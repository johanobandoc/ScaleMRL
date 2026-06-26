import torch
import torch.nn as nn

from common import layers, math, init
from tensordict import TensorDict


class WorldModel(nn.Module):
	"""
	TD-MPC2 self-predictive world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		if cfg.finetune:
			self._task_emb = nn.Embedding(200, cfg.task_dim)
			self._task_emb._parameters['weight'] = torch.tensor(self.cfg.task_embeddings[:1], dtype=torch.float32).repeat(200, 1)
			print(f'Using pre-computed language embeddings for task {self.cfg.task}.')
		else:
			self._task_emb = nn.Embedding(len(cfg.task_embeddings), cfg.task_dim) if cfg.task_dim > 0 else None
			if cfg.multitask:
				self._task_emb._parameters['weight'] = torch.tensor(self.cfg.task_embeddings, dtype=torch.float32)
				if cfg.rank == 0:
					print('Using pre-computed language embeddings.')
		if self._task_emb is not None:
			self._task_emb.weight.requires_grad = False  # Freeze task embeddings
		if cfg.finetune:
			self.register_buffer("_action_masks", torch.zeros(200, cfg.action_dim))
			self._action_masks[:, :cfg.action_dims[0]] = 1.
		else:
			self.register_buffer("_action_masks", torch.zeros(len(cfg.action_dims), cfg.action_dim))
			for i in range(len(cfg.action_dims)):
				self._action_masks[i, :cfg.action_dims[i]] = 1.
		self._encoder = layers.enc(cfg)
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self._reward = layers.mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = layers.mlp(cfg.latent_dim + cfg.task_dim, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = layers.QOnlineTargetEnsemble(cfg)
		self.apply(init.weight_init)
		init.zero_(self._reward[-1].weight)
		for i in range(cfg.num_q):
			init.zero_(self._Qs.online._Qs[i][-1].weight)
			init.zero_(self._Qs.target._Qs[i][-1].weight)
		self._Qs.hard_update_target()
		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)

	def __repr__(self):
		repr = 'Newt World Model\n'
		modules = ['Encoder', 'Dynamics', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._dynamics, self._reward, self._pi, self._Qs.online]):
			params = "{:,}".format(sum(p.numel() for p in m.parameters() if p.requires_grad))
			repr += f"{modules[i]} ({params}): {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._Qs.target.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._Qs.soft_update_target()

	def task_emb(self, x, task):
		"""
		Appends task embedding to input x along the last dimension.
		Handles broadcast, reshape, and shape mismatches robustly.
		"""
		if not hasattr(self, '_task_emb') or self._task_emb is None:
			return x

		if isinstance(task, int):
			task = torch.tensor([task], device=x.device)

		x_batch_shape = x.shape[:-1]
		E = self._task_emb.embedding_dim

		# Step 1: Pad task shape (add singleton dims) until it's same rank as x_batch_shape
		while task.ndim < len(x_batch_shape):
			task = task.unsqueeze(-1)

		# Step 2: Try broadcasting
		try:
			broadcast_shape = torch.broadcast_shapes(task.shape, x_batch_shape)
		except RuntimeError:
			# Step 3: Try reshape fallback if total number of elements match
			if task.numel() == int(torch.tensor(x_batch_shape).prod().item()):
				task = task.reshape(*x_batch_shape)
			else:
				raise ValueError(
					f"Incompatible task shape: got {task.shape}, expected broadcastable to {x_batch_shape} "
					f"(x.shape = {x.shape})"
				)

		# Step 4: Embed and expand
		emb = self._task_emb(task.long())  # shape (..., E)
		while emb.ndim < x.ndim:
			emb = emb.unsqueeze(-2)

		emb = emb.expand(*x_batch_shape, E)
		return torch.cat([x, emb], dim=-1)

	def encode(self, obs, task):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		if self.cfg.obs == 'state':
			return self._encoder[self.cfg.obs](self.task_emb(obs, task))
		assert isinstance(obs, TensorDict), "Expected observation to be a TensorDict"
		z = torch.cat([self.task_emb(obs['state'], task), obs['rgb']], dim=-1)
		return self._encoder['state'](z)

		# z_rgb = self._encoder['rgb'](obs['rgb'])
		# return torch.stack((z_state, z_rgb), dim=0).mean(0)
		
		# z_state = self._encoder['state'](self.task_emb(obs['state'], task))
		# z_cat = torch.cat([z_state, self.task_emb(obs['rgb'], task)], dim=-1)
		# out = self._encoder['rgb'](z_cat)
		
		return out

	def next(self, z, a, task):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)

	def reward(self, z, a, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)
		return self._reward(z)
	
	def pi(self, z, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		z = self.task_emb(z, task)

		# Gaussian policy prior
		mean, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mean)

		action_mask = self._action_masks[task]  # shape: (*batch_dims, action_dim)
		while action_mask.ndim < mean.ndim:
			action_mask = action_mask.unsqueeze(-2)  # Add sequence dim (or other mid-batch dim)
		action_mask = action_mask.expand_as(mean)  # Ensure shape matches mean

		mean = mean * action_mask
		log_std = log_std * action_mask
		eps = eps * action_mask

		action_dims = action_mask.sum(-1, keepdim=True)
		log_prob = math.gaussian_logprob(eps, log_std)

		# Scale log probability by action dimensions
		size = eps.shape[-1] if action_dims is None else action_dims
		scaled_log_prob = log_prob * size

		# Reparameterization trick
		action = mean + eps * log_std.exp()
		mean, action, log_prob = math.squash(mean, action, log_prob)

		entropy_scale = scaled_log_prob / (log_prob + 1e-8)
		info = TensorDict({
			"mean": mean,
			"log_std": log_std,
			"entropy": -log_prob,
			"scaled_entropy": -log_prob * entropy_scale,
		})
		return action, info

	def Q(self, z, a, task, return_type='min', target=False, detach=False):
		"""
		Predict state-action value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled Q-values.
			- `avg`: return the average of two randomly subsampled Q-values.
			- `all`: return all Q-values.
		`target` specifies whether to use the target Q-networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}
		z = self.task_emb(z, task)
		z = torch.cat([z, a], dim=-1)

		out = self._Qs(z, target=target)
		if detach:
			out = out.detach()

		if return_type == 'all':
			return out

		qidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		Q = math.two_hot_inv(out[qidx], self.cfg)
		if return_type == "min":
			return Q.min(0).values
		return Q.sum(0) / 2
