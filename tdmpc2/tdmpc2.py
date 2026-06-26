from copy import deepcopy

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from common import math
from common.scale import RunningScale
from common.layers import api_model_conversion


class TDMPC2(torch.nn.Module):
	"""
	Newt-based TD-MPC2 agent. Implements training + inference.
	Can be used for both single-task and multi-task experiments,
	and supports both state and pixel observations.
	"""

	def __init__(self, model, cfg):
		super().__init__()
		self.cfg = deepcopy(cfg)
		self.cfg.action_dim = cfg.action_dim
		self.device = torch.device(f'cuda:{self.cfg.rank}')
		self.model = model
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.online.parameters()},
			{'params': self.model._pi.parameters()},
		], lr=self.cfg.lr, fused=True, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, fused=True, capturable=True)
		if self.cfg.lr_schedule:
			self.scheduler = math.MultiWarmupConstantLR(
				[self.optim, self.pi_optim],
				warmup_steps=self.cfg.warmup_steps,
			)
			if self.cfg.rank == 0:
				print(f'Using {self.cfg.lr_schedule} learning rate schedule with {self.cfg.warmup_steps} warmup steps.')
		elif self.cfg.rank == 0:
			print('No learning rate schedule specified, using constant LR.')
		self.model.eval()
		self.maxq_pi = True
		self.scale = RunningScale(self.cfg)
		self.discount = torch.tensor(self.cfg.discounts, device=self.device, dtype=torch.float32)
		if self.cfg.rank == 0:
			print('Episode length:', self.cfg.episode_length)
			print('Discount factor:', self.discount)
		self._prev_mean = torch.zeros(self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim, device=self.device)
		self.rho = torch.pow(self.cfg.rho, torch.arange(self.cfg.horizon+1, device=self.device))
		self.rho = self.rho / self.rho.sum()

		# Compile methods for faster training/inference
		if self.compile and self.cfg.rank == 0:
			print('Compiling methods...')
		self.pi = self._maybe_compile(self._pi)
		self.sample_pi_trajs = self._maybe_compile(self._sample_pi_trajs)
		self.mppi = self._maybe_compile(self._mppi)
		self.pi_loss = self._maybe_compile(self._pi_loss)
		self.loss_fn = self._maybe_compile(self._loss_fn)

	def _maybe_compile(self, fn):
		return torch.compile(fn, mode="reduce-overhead") if self.cfg.compile else fn

	def save(self, fp):
		"""
		Save state dict of the agent to filepath.

		Args:
			fp (str): Filepath to save state dict to.
		"""
		torch.save({
			"model": self.model.state_dict(),
			"optim": self.optim.state_dict(),
			"pi_optim": self.pi_optim.state_dict(),
			"scale": self.scale.state_dict(),
		}, fp)

	def load(self, fp):
		"""
		Load a saved state dict from filepath (or dictionary) into current agent.

		Args:
			fp (str): Filepath to load state dict from.
		"""
		state_dict = torch.load(fp, map_location=torch.get_default_device(), weights_only=False)
		state_dict = state_dict["model"] if "model" in state_dict else state_dict
		
		# Retain task_emb and action_masks if finetuning
		if self.cfg.finetune:
			prefix = "module." if "module._task_emb.weight" in state_dict else ""
			state_dict[prefix+"_task_emb.weight"] = self.model._task_emb.weight
			state_dict[prefix+"_action_masks"] = self.model._action_masks

		state_dict = api_model_conversion(self.model.state_dict(), state_dict)
		self.model.load_state_dict(state_dict)

	@torch.no_grad()
	def _pi(self, obs, task=None):
		"""
		Select an action using the policy network.
		"""
		z = self.model.encode(obs, task)
		action, info = self.model.pi(z, task)
		return action, info

	@torch.no_grad()
	def forward(self, obs, t0, step=None, eval_mode=False, task=None, mpc=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.Tensor): Whether this is the first observation in the episode.
			step (int): Current environment step.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index.
			mpc (bool): Whether to use model predictive control.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		if isinstance(obs, dict):
			obs = TensorDict(obs)
		obs = obs.to(self.device, non_blocking=True)
		if task is not None and not isinstance(task, torch.Tensor):
			task = torch.tensor([task], device=self.device)
		if task is not None and task.device != self.device:
			task = task.to(self.device, non_blocking=True)
		mpc = mpc if mpc is not None else self.cfg.mpc
		if mpc:
			if t0.device != self.device:
				t0 = t0.to(self.device, non_blocking=True)
			action = self.plan(obs, t0=t0, step=step, eval_mode=eval_mode, task=task)
		else:
			action, info = self.pi(obs, task)
			if eval_mode:
				action = info["mean"]
		return action.cpu()
	
	@torch.no_grad()
	def _estimate_value(self, z, actions, task):
		"""Estimate value of a trajectory starting at latent state z and executing given actions."""
		G = torch.zeros(self.cfg.num_envs, self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		discount = torch.ones(self.cfg.num_envs, self.cfg.num_samples, 1, dtype=torch.float32, device=z.device)
		for t in range(self.cfg.horizon):
			reward = math.two_hot_inv(self.model.reward(z, actions[:, t], task), self.cfg)
			z = self.model.next(z, actions[:, t], task)
			G = G + discount * reward
			discount_update = self.discount[task].view(-1, 1, 1)
			discount = discount * discount_update
		action, _ = self.model.pi(z, task)
		value = self.model.Q(z, action, task, return_type='avg')
		return G + discount * value
	
	@torch.no_grad()
	def _sample_pi_trajs(self, obs, task=None):
		z = self.model.encode(obs, task)
		pi_actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
		_z = z.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs, 1).view(self.cfg.num_envs * self.cfg.num_pi_trajs, -1)
		_task = task.unsqueeze(1).repeat(1, self.cfg.num_pi_trajs).view(self.cfg.num_envs * self.cfg.num_pi_trajs)
		for t in range(self.cfg.horizon - 1):
			a, _ = self.model.pi(_z, _task)
			pi_actions[:, t] = a.view(self.cfg.num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
			_z = self.model.next(_z, a, _task)
		a, _ = self.model.pi(_z, _task)
		pi_actions[:, -1] = a.view(self.cfg.num_envs, self.cfg.num_pi_trajs, self.cfg.action_dim)
		return pi_actions, z
	
	@torch.no_grad()
	def _mppi(self, z, pi_actions, task, mean, std):
		"""
		MPPI loop.
		"""
		actions = torch.empty(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :, :self.cfg.num_pi_trajs] = pi_actions
		action_mask = self.model._action_masks[task].unsqueeze(1).unsqueeze(1)

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample new actions
			r = torch.randn(self.cfg.num_envs, self.cfg.horizon, self.cfg.num_samples - self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(2) + std.unsqueeze(2) * r
			actions[:, :, self.cfg.num_pi_trajs:] = actions_sample.clamp(-1, 1)
			actions = actions * action_mask

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(2), self.cfg.num_elites, dim=1).indices
			elite_value = torch.gather(value, 1, elite_idxs.unsqueeze(2))
			elite_actions = actions.gather(
				dim=2,
				index=elite_idxs[:, None, :, None].expand(-1, self.cfg.horizon, self.cfg.num_elites, self.cfg.action_dim)
			)

			# Update parameters
			score = torch.exp(self.cfg.temperature * (elite_value - elite_value.max(1, keepdim=True).values))
			score = score / (score.sum(dim=1, keepdim=True) + 1e-9)
			score_exp = score.unsqueeze(1)
			mean = (score_exp * elite_actions).sum(dim=2) / (score_exp.sum(dim=2) + 1e-9)
			std = ((score_exp * (elite_actions - mean.unsqueeze(2)) ** 2).sum(dim=2) /
				(score_exp.sum(dim=2) + 1e-9)).sqrt().clamp(self.cfg.min_std, self.cfg.max_std)
			mean = mean * action_mask.squeeze(2)
			std = std * action_mask.squeeze(2)

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(2), temperature=self.cfg.temperature, dim=1)
		selected_actions = elite_actions.gather(
			dim=2,
			index=rand_idx[:, None, None, None].expand(-1, self.cfg.horizon, 1, self.cfg.action_dim)
		).squeeze(2)
		action, std_out = selected_actions[:, 0], std[:, 0]

		return action.clamp(-1, 1), mean, std_out

	@torch.no_grad()
	def plan(self, obs, t0, step=None, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			obs (torch.Tensor): Observation from the environment.
			t0 (torch.Tensor): Whether this is the first observation in the episode.
			step (int): Current environment step.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (torch.Tensor): Task index.

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		pi_actions, z = self.sample_pi_trajs(obs, task)

		# Initialize state and parameters
		z = z.unsqueeze(1).repeat(1, self.cfg.num_samples, 1)
		shifted = torch.cat([self._prev_mean[:, 1:], torch.zeros_like(self._prev_mean[:, :1])], dim=1)
		mean = torch.where(t0.view(self._prev_mean.shape[0], 1, 1), torch.zeros_like(shifted), shifted)
		std = torch.full((self.cfg.num_envs, self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, device=self.device)

		if self.cfg.constrained_planning and step is not None:
			# Init planning with policy statistics
			pi_mean = pi_actions.mean(2)
			pi_std = pi_actions.std(2).clamp(self.cfg.min_std, self.cfg.max_std)
			mean, std = math.interp_dist(mean, std, pi_mean, pi_std, step,
				self.cfg.constraint_start_step, self.cfg.constraint_final_step)

		# Optimize with MPPI
		action, out_mean, out_std = self.mppi(z, pi_actions, task, mean, std)
		self._prev_mean = out_mean.clone()
		if not eval_mode:
			action = (action + out_std * torch.randn_like(action)).clamp(-1, 1)

		return action
	
	def _pi_loss(self, zs, action, task):
		"""Compute the policy loss."""
		pi_action, info = self.model.pi(zs, task)

		# Policy prior loss
		pi_prior_loss = (math.masked_bc_per_timestep(pi_action[:-1], action, task, self.model._action_masks) \
				   * self.rho[:-1, None]).sum(0)

		# Normalized Q-loss
		qs = self.model.Q(zs, pi_action, task, return_type='avg')
		scaled_qs = self.scale(qs)
		maxq_loss = ((-self.cfg.entropy_coef*info["scaled_entropy"] - scaled_qs) * self.rho[:, None, None]).sum(dim=(0,2))
		
		# Compute total policy loss
		pi_loss = (pi_prior_loss + maxq_loss).mean()

		info = TensorDict({
			"pi_prior_loss": pi_prior_loss.mean(),
			"pi_loss": pi_loss,
			"pi_entropy": info["entropy"],
			"pi_scaled_entropy": info["scaled_entropy"],
			"pi_std": info["log_std"].exp().mean(),
			"pi_max_std": info["log_std"].exp().max(),
		})
		return pi_loss, qs[0].detach(), info
	
	def update_pi(self, zs, action, task):
		"""
		Update policy using a sequence of latent states.

		Args:
			zs (torch.Tensor): Sequence of latent states.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			float: Loss of the policy update.
		"""
		self.model._Qs.track_grad(False)

		# Compute policy loss
		pi_loss, qs, info = self.pi_loss(zs, action, task)
		
		# Update policy
		pi_loss.backward()
		pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
		self.pi_optim.step()
		self.pi_optim.zero_grad(set_to_none=True)
		self.model._Qs.track_grad(True)

		# Update running statistics
		with torch.no_grad():
			self.scale.update(qs)  # local update
			if torch.distributed.is_initialized():
				torch.distributed.all_reduce(self.scale.value, op=torch.distributed.ReduceOp.SUM)
				self.scale.value.div_(self.cfg.world_size)

		info.update({
			"pi_grad_norm": pi_grad_norm,
			"pi_scale": self.scale.value,
		})
		return info

	@torch.no_grad()
	def _td_target(self, next_z, reward, task):
		"""
		Compute the TD-target from a reward and the observation at the following time step.

		Args:
			next_z (torch.Tensor): Latent state at the following time step.
			reward (torch.Tensor): Reward at the current time step.
			task (torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: TD-target.
		"""
		action, _ = self.model.pi(next_z, task)
		discount = self.discount[task].unsqueeze(-1)
		return reward + discount * self.model.Q(next_z, action, task, return_type='min', target=True)

	def _loss_fn(self, obs, action, reward, task=None):
		"""
		Compute the model loss for a batch of data.
		"""
		# Compute targets
		with torch.no_grad():
			next_z = self.model.encode(obs[1:], task)
			td_targets = self._td_target(next_z, reward, task)

		# Latent rollout
		zs = torch.empty(self.cfg.horizon+1, self.cfg.batch_size, self.cfg.latent_dim, device=self.device)
		z = self.model.encode(obs[0], task[0])
		zs[0] = z
		consistency_loss = 0
		for t, (_action, _next_z, _task) in enumerate(zip(action.unbind(0), next_z.unbind(0), task.unbind(0))):
			z = self.model.next(z, _action, _task)
			consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.rho[t]
			zs[t+1] = z

		# Predictions
		_zs = zs[:-1]
		qs = self.model.Q(_zs, action, task, return_type='all')
		reward_preds = self.model.reward(_zs, action, task)

		# Compute losses
		reward_loss, value_loss = 0, 0
		for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))):
			reward_loss = reward_loss + math.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.rho[t]
			for _, qs_unbind_unbind in enumerate(qs_unbind.unbind(0)):
				value_loss = value_loss + math.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.rho[t]
		value_loss = value_loss / self.cfg.num_q

		if not self.maxq_pi: # Behavior cloning
			pi_action, pi_info = self.model.pi(_zs, task)
			bc_loss = math.masked_bc_per_timestep(pi_action, action, task, self.model._action_masks)
			entropy_loss = -self.cfg.entropy_coef*pi_info["scaled_entropy"].squeeze(-1)
			pi_prior_loss = ((bc_loss + entropy_loss) * self.rho[:-1, None]).mean()
			pi_info = TensorDict({
				"bc_loss": bc_loss,
				"entropy_loss": entropy_loss,
				"pi_prior_loss": pi_prior_loss,
				"pi_entropy": pi_info["entropy"],
				"pi_scaled_entropy": pi_info["scaled_entropy"],
				"pi_std": pi_info["log_std"].exp().mean(),
				"pi_max_std": pi_info["log_std"].exp().max(),
			})
		else:
			pi_prior_loss = 0
			pi_info = TensorDict({})

		total_loss = (
			self.cfg.consistency_coef * consistency_loss +
			self.cfg.reward_coef * reward_loss +
			self.cfg.value_coef * value_loss +
			self.cfg.prior_coef * pi_prior_loss
		)

		info = TensorDict({
			"consistency_loss": consistency_loss,
			"reward_loss": reward_loss,
			"value_loss": value_loss,
			"total_loss": total_loss,
		})
		info.update(pi_info)

		return total_loss, zs.detach(), info.detach()

	def update(self, buffer):
		"""
		Main update function. Corresponds to one iteration of model learning.

		Args:
			buffer (common.buffer.Buffer): Replay buffer.

		Returns:
			dict: Dictionary of training statistics.
		"""
		obs, action, reward, task = buffer.sample(device=self.device)

		# Prepare for update
		self.model.train()

		# Step the learning rate scheduler
		if self.cfg.lr_schedule:
			self.scheduler.step()

		# Compute loss
		torch.compiler.cudagraph_mark_step_begin()
		total_loss, zs, info = self.loss_fn(obs, action, reward, task)

		# Update model
		total_loss.backward()
		grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
		self.optim.step()
		self.optim.zero_grad(set_to_none=True)

		# Update target Q-functions
		self.model.soft_update_target_Q()

		# Max-Q policy update
		if self.maxq_pi:
			pi_info = self.update_pi(zs, action, task[:1])
			info.update(pi_info)
		
		# Return training statistics
		self.model.eval()
		info.update({
			"grad_norm": grad_norm,
		})
		if self.cfg.lr_schedule:
			info.update({
				"lr_enc": self.scheduler.current_lr(0, 0),
				"lr": self.scheduler.current_lr(0, 1),
				"lr_pi": self.scheduler.current_lr(1, 0),
			})
		return info.detach().mean()
