"""
MR.Q agent reimplemented on top of the mmbench / TD-MPC2 infrastructure.

Architecture
------------
MRQModel  (mirrors WorldModel's public interface)
  _task_emb       – frozen CLIP task embeddings  (identical to WorldModel)
  _action_masks   – per-task action masks        (identical to WorldModel)
  _encoder        – (obs ‖ task_emb) → zs  [latent_dim]   reuses layers.enc()
  _za             – action → za  [mrq_za_dim]              MRQ-specific
  _zsa            – (zs ‖ za) → zsa  [latent_dim]          reuses layers.mlp()
  _dynamics_model – zsa → (next_zs ‖ reward_logits)        MRQ-specific linear head
  _pi             – (zs ‖ task_emb) → action  [action_dim] reuses layers.policy()
  _Qs             – mrq_num_q × (zsa → Q [1 scalar | num_bins logits])  reuses layers.mlp()
  + EMA target copies of _encoder / _za / _zsa / _pi / _Qs

MRQ  (mirrors TDMPC2 agent interface)
  forward(obs, t0, step, eval_mode, task, mpc)
             – direct policy + TD3 exploration noise; mpc flag ignored
  update(buffer) – three sequential gradient steps (see below) + soft target update
  save(fp) / load(fp) – single .pt checkpoint (same convention as TDMPC2)

Gradient isolation in update()
--------------------------------
Three separate AdamW optimizers; each step only touches its own parameter group:
  Phase 1 – encoder_optim  →  {_encoder, _za, _zsa, _dynamics_model}
  Phase 2 – value_optim    →  {_Qs}               (encoder used inside no_grad)
  Phase 3 – policy_optim   →  {_pi}               (encoder + Q frozen before forward)

Config fields consumed from the shared TDMPC2 Config
------------------------------------------------------
  latent_dim, mlp_dim, enc_dim, num_enc_layers, simnorm_dim  – architecture
  task_dim, task_embeddings, action_dims, action_dim          – task/env
  obs_shape, obs                                              – observation
  discounts                                                   – per-task γ
  lr, enc_lr_scale, grad_clip_norm, tau                       – optimisation
  horizon, num_bins, vmin, vmax, bin_size                     – shared hypers

MRQ-specific config fields  (cfg.mrq_*)
----------------------------------------
  mrq_za_dim              – action embedding dim for _za
  mrq_exploration_noise   – TD3 online noise std (σ_explore)
  mrq_target_policy_noise – TD3 target smoothing noise std (σ_target)
  mrq_noise_clip          – TD3 noise clipping bound
  mrq_pre_activ_weight    – pre-activation L2 regularisation weight
  mrq_dyn_weight          – dynamics loss coefficient in encoder loss
  mrq_reward_weight       – reward loss coefficient in encoder loss
  mrq_enc_horizon         – encoder unroll steps (default 5, paper value);
                            capped to cfg.horizon at runtime
  mrq_policy_layernorm    – if True: policy uses LayerNorm+Mish hidden layers
                            (closer to original MRQ BaseMLP);
                            if False (default): bare ReLU, no normalisation
  mrq_distributional_q    – if True: Q outputs num_bins logits + soft-CE loss
                            (like NEWT); if False (default): scalar Q + smooth-L1
  mrq_reward_norm         – reward/Q normalisation: "none" (default) |
                            "mrq" (mean-abs reward EMA on Q-targets) |
                            "newt" (RunningScale IQR on Q in policy loss)
  mrq_num_q               – number of Q-networks (default 2 = original MRQ
                            twin-Q); ignored when mrq_num_q_from_model_size=True
  mrq_num_q_from_model_size – if True: use cfg.num_q driven by model_size
                            (S→3, B/L→5, XL→7), same scaling as NEWT;
                            if False (default): always use mrq_num_q=2.
                            Value loss trains all; Q_target subsamples 2 (min);
                            policy uses Q_0 only if num_q==2 (TD3), else avg
                            of random 2 (NEWT-style)

Key differences from original MR.Q paper
------------------------------------------
* Multi-task: task-conditioned via frozen CLIP embeddings concatenated to obs.
* Buffer: uses mmbench's torchrl ReplayBuffer (no separate LAP replay buffer).
* Target updates: soft EMA every step (tau = cfg.tau) vs. periodic hard resets.
* Horizon: encoder unrolling uses cfg.horizon steps (= buffer horizon).
* Q-values: scalar smooth-L1 by default (faithful to original MR.Q);
*           distributional (num_bins logits + soft-CE) if mrq_distributional_q=True.
"""

from copy import deepcopy
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import layers, math, init
from common.scale import RunningScale


# ======================================================================
# Model
# ======================================================================

class MRQModel(nn.Module):
	"""
	Neural network components for MR.Q.
	Designed to mirror WorldModel's public interface so the same Trainer,
	Logger, and train.py plumbing work unchanged.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg

		# ---- Task embedding (frozen CLIP, same as TDMPC2) ---------------
		if cfg.task_dim > 0:
			self._task_emb = nn.Embedding(
				len(cfg.task_embeddings), cfg.task_dim)
			self._task_emb._parameters['weight'] = torch.tensor(
				cfg.task_embeddings, dtype=torch.float32)
			self._task_emb.weight.requires_grad = False
		else:
			self._task_emb = None

		# ---- Action masks (same as TDMPC2) --------------------------------
		self.register_buffer(
			'_action_masks',
			torch.zeros(len(cfg.action_dims), cfg.action_dim))
		for i, d in enumerate(cfg.action_dims):
			self._action_masks[i, :d] = 1.0

		# ---- State encoder: (obs ‖ task_emb) → zs -----------------------
		# Reuses TDMPC2's enc() which builds NormedLinear + SimNorm MLP.
		# Input dim = obs_dim + task_dim  (task_dim=0 for single-task).
		self._encoder = layers.enc(cfg)

		# ---- Action embedding: action → za --------------------------------
		self._za = nn.Sequential(
			nn.Linear(cfg.action_dim, cfg.mrq_za_dim),
			nn.ELU(),
		)

		# ---- Joint encoder: (zs ‖ za) → zsa ------------------------------
		self._zsa = layers.mlp(
			cfg.latent_dim + cfg.mrq_za_dim,
			2 * [cfg.mlp_dim],
			cfg.latent_dim,
		)

		# ---- Auxiliary head: zsa → (next_zs ‖ reward_logits) ------------
		self._dynamics_model = nn.Linear(
			cfg.latent_dim,
			cfg.latent_dim + max(cfg.num_bins, 1),
		)

		# ---- Policy: (zs ‖ task_emb) → pre-tanh action ------------------
		# mrq_policy_layernorm=False (default): bare ReLU MLP, no normalisation.
		# mrq_policy_layernorm=True:            LayerNorm+Mish hidden layers
		#   (closer to original MRQ's BaseMLP which uses LayerNorm+ELU).
		pi_in = cfg.latent_dim + cfg.task_dim
		if getattr(cfg, 'mrq_policy_layernorm', False):
			self._pi = layers.mlp(pi_in, 2 * [cfg.mlp_dim], cfg.action_dim)
		else:
			self._pi = layers.policy(pi_in, 2 * [cfg.mlp_dim], cfg.action_dim)

		# ---- Q ensemble ------------------------------------------------------
		# mrq_distributional_q=False (default): scalar output (dim=1), smooth-L1 loss.
		# mrq_distributional_q=True:            distributional output (num_bins logits),
		#                                        soft-CE loss (same as NEWT).
		# mrq_num_q: number of Q-networks; value loss trains all, policy/target
		#            randomly subsample 2 (NEWT-style).
		q_out_dim = max(cfg.num_bins, 1) if getattr(cfg, 'mrq_distributional_q', False) else 1
		if getattr(cfg, 'mrq_num_q_from_model_size', False):
			num_q = cfg.num_q  # follows MODEL_SIZE: S→3, B/L→5, XL→7
		else:
			num_q = getattr(cfg, 'mrq_num_q', 2)  # fixed, default=2 (original MRQ)
		self._Qs = nn.ModuleList([
			layers.mlp(cfg.latent_dim, 2 * [cfg.mlp_dim], q_out_dim)
			for _ in range(num_q)
		])

		# ---- Weight initialisation BEFORE deepcopy so targets start identically
		self.apply(init.weight_init)

		# ---- Target networks (copies of already-initialised online nets) ---
		self._encoder_target = deepcopy(self._encoder)
		self._za_target      = deepcopy(self._za)
		self._zsa_target     = deepcopy(self._zsa)
		self._pi_target      = deepcopy(self._pi)
		self._Qs_target      = deepcopy(self._Qs)
		for p in self._target_params():
			p.requires_grad_(False)

	# ------------------------------------------------------------------
	# Parameter groups
	# ------------------------------------------------------------------

	def encoder_params(self):
		"""Online encoder parameters (for encoder optimizer)."""
		return chain(
			self._encoder.parameters(),
			self._za.parameters(),
			self._zsa.parameters(),
			self._dynamics_model.parameters(),
		)

	def value_params(self):
		"""Online Q-network parameters (for value optimizer)."""
		return self._Qs.parameters()

	def policy_params(self):
		"""Online policy parameters (for policy optimizer)."""
		return self._pi.parameters()

	def _target_params(self):
		return chain(
			self._encoder_target.parameters(),
			self._za_target.parameters(),
			self._zsa_target.parameters(),
			self._pi_target.parameters(),
			self._Qs_target.parameters(),
		)

	# ------------------------------------------------------------------
	# Forward methods
	# ------------------------------------------------------------------

	def _task_emb_for(self, task):
		"""Return task embedding for a task tensor, or None for single-task."""
		if self._task_emb is None or task is None:
			return None
		return self._task_emb(task)

	def _build_encoder_input(self, obs, task_emb):
		"""Build encoder input: [state, task_emb, rgb(optional)] matching layers.enc order."""
		is_dict = isinstance(obs, dict) or hasattr(obs, 'get')
		x = obs['state'] if is_dict else obs
		parts = [x]
		if task_emb is not None:
			parts.append(task_emb)
		if self.cfg.obs == 'rgb' and is_dict:
			parts.append(obs['rgb'])
		return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

	def encode(self, obs, task=None):
		"""Online encoder: obs → zs."""
		emb = self._task_emb_for(task)
		inp = self._build_encoder_input(obs, emb)
		return self._encoder['state'](inp)

	def encode_target(self, obs, task=None):
		"""Target encoder: obs → zs (no gradient)."""
		emb = self._task_emb_for(task)
		inp = self._build_encoder_input(obs, emb)
		return self._encoder_target['state'](inp)

	def encode_zsa(self, zs, action):
		"""Online joint encoder: (zs, action) → zsa."""
		return self._zsa(torch.cat([zs, self._za(action)], dim=-1))

	def encode_zsa_target(self, zs, action):
		"""Target joint encoder: (zs, action) → zsa (no gradient)."""
		return self._zsa_target(torch.cat([zs, self._za_target(action)], dim=-1))

	def predict(self, zs, action):
		"""Auxiliary prediction: (zs, action) → (pred_next_zs, pred_reward_logits)."""
		zsa = self.encode_zsa(zs, action)
		out = self._dynamics_model(zsa)
		return out[:, :self.cfg.latent_dim], out[:, self.cfg.latent_dim:]

	def pi(self, zs, task=None):
		"""Online policy: zs → (action, pre_activ).  action = tanh(pre_activ)."""
		emb = self._task_emb_for(task)
		inp = torch.cat([zs, emb], dim=-1) if emb is not None else zs
		pre = self._pi(inp)
		return torch.tanh(pre), pre

	def pi_target(self, zs, task=None):
		"""Target policy: zs → action (tanh applied)."""
		emb = self._task_emb_for(task)
		inp = torch.cat([zs, emb], dim=-1) if emb is not None else zs
		return torch.tanh(self._pi_target(inp))

	def Q(self, zsa):
		"""Online Q ensemble: randomly subsample 2 networks → scalars cat along dim=1 → (B, 2)."""
		num_q = len(self._Qs)
		idx = torch.randperm(num_q, device=zsa.device)[:2]
		qs = []
		for i in idx:
			q = self._Qs[i](zsa)
			if getattr(self.cfg, 'mrq_distributional_q', False):
				q = math.two_hot_inv(q, self.cfg)
			qs.append(q)
		return torch.cat(qs, dim=1)

	def Q_target(self, zsa):
		"""Target Q ensemble: randomly subsample 2 → min as scalar → (B, 1)."""
		num_q = len(self._Qs_target)
		idx = torch.randperm(num_q, device=zsa.device)[:2]
		qs = []
		for i in idx:
			q = self._Qs_target[i](zsa)
			if getattr(self.cfg, 'mrq_distributional_q', False):
				q = math.two_hot_inv(q, self.cfg)
			qs.append(q)
		return torch.min(qs[0], qs[1])

	# ------------------------------------------------------------------
	# Target network update
	# ------------------------------------------------------------------

	@torch.no_grad()
	def soft_update_targets(self):
		"""EMA soft update of all target networks."""
		tau = self.cfg.tau
		pairs = [
			(self._encoder_target, self._encoder),
			(self._za_target,      self._za),
			(self._zsa_target,     self._zsa),
			(self._pi_target,      self._pi),
		]
		for target, online in pairs:
			for tp, op in zip(target.parameters(), online.parameters()):
				tp.data.lerp_(op.data, tau)
		for q_target, q_online in zip(self._Qs_target, self._Qs):
			for tp, op in zip(q_target.parameters(), q_online.parameters()):
				tp.data.lerp_(op.data, tau)

	# ------------------------------------------------------------------
	# Printable summary (Trainer prints self.agent.model at startup)
	# ------------------------------------------------------------------

	def __repr__(self):
		enc = self._encoder['state']
		n_enc  = sum(p.numel() for p in self._encoder.parameters())
		n_zsa  = sum(p.numel() for p in chain(self._za.parameters(),
											   self._zsa.parameters(),
											   self._dynamics_model.parameters()))
		n_pi   = sum(p.numel() for p in self._pi.parameters())
		num_q  = len(self._Qs)
		n_q    = sum(p.numel() for p in self._Qs.parameters())
		total  = n_enc + n_zsa + n_pi + n_q
		q_out  = 'num_bins' if getattr(self.cfg, 'mrq_distributional_q', False) else '1'
		return (
			f'MRQModel(\n'
			f'  encoder:         {n_enc:>10,} params  '
			f'({self.cfg.obs_shape["state"][0]}+{self.cfg.task_dim}→{self.cfg.latent_dim})\n'
			f'  joint_enc+dyn:   {n_zsa:>10,} params  '
			f'(zs+za→zsa→next_zs+reward)\n'
			f'  policy:          {n_pi:>10,} params  '
			f'({self.cfg.latent_dim}+{self.cfg.task_dim}→{self.cfg.action_dim})\n'
			f'  Q_ensemble:      {n_q:>10,} params  '
			f'(zsa→{q_out} ×{num_q})\n'
			f'  total (online):  {total:>10,} params\n'
			f')'
		)


# ======================================================================
# Agent
# ======================================================================

class MRQ(nn.Module):
	"""
	MR.Q agent.  Mirrors the TDMPC2 interface so the existing Trainer,
	Logger, and train.py work without modification.

	Action selection
	----------------
	Direct policy output + TD3 exploration noise (no MPC planning).
	`mpc` flag is accepted but silently ignored.

	Training  (one call to update() per gradient step)
	--------------------------------------------------
	1. Encoder auxiliary loss  – unroll dynamics for `cfg.horizon` steps,
	   minimise MSE(pred_next_zs, target_next_zs) and soft-CE reward loss.
	2. Value (TD3) loss        – Smooth-L1 between online Q(zsa) and the
	   multi-step Bellman target (frozen encoder).
	3. Policy loss             – maximise Q1(zsa(zs, pi(zs))) subject to a
	   small pre-activation regulariser (frozen encoder & Q).
	4. Soft EMA target update.
	"""

	def __init__(self, model: MRQModel, cfg):
		super().__init__()
		self.cfg   = cfg
		self.model = model
		self.device = torch.device(f'cuda:{cfg.rank}')

		# ---- Optimisers ---------------------------------------------------
		self.encoder_optim = torch.optim.AdamW(
			model.encoder_params(),
			lr=cfg.enc_lr_scale * cfg.lr,
			weight_decay=1e-4,
		)
		self.value_optim = torch.optim.AdamW(
			model.value_params(),
			lr=cfg.lr,
			weight_decay=1e-4,
		)
		self.policy_optim = torch.optim.AdamW(
			model.policy_params(),
			lr=cfg.lr,
			weight_decay=1e-4,
		)

		# Per-task discount factors (same convention as TDMPC2)
		self.discount = torch.tensor(
			cfg.discounts, device=self.device, dtype=torch.float32)

		# Reward / Q normalisation
		norm = getattr(cfg, 'mrq_reward_norm', 'none')
		if norm == 'newt':
			# Normalise Q-values in the policy loss (mirrors TDMPC2 RunningScale).
			self._q_scale = RunningScale(cfg)
		elif norm == 'mrq':
			# Normalise Q-targets by a running scale of batch reward magnitudes
			# (mirrors original MR.Q reward_scale / target_reward_scale).
			self.register_buffer('_reward_scale',
				torch.ones(1, device=self.device))
			self.register_buffer('_target_reward_scale',
				torch.ones(1, device=self.device))

		self.model.eval()

	# ------------------------------------------------------------------
	# Trainer interface
	# ------------------------------------------------------------------

	@torch.no_grad()
	def forward(self, obs, t0=None, step=None, eval_mode=False,
				task=None, mpc=None):
		"""
		Select actions for all parallel envs (vectorised).

		Args:
			obs:       (num_envs, obs_dim) tensor or TensorDict with 'state'.
			task:      (num_envs,) int task IDs.
			eval_mode: disable exploration noise.
		Returns:
			action tensor (num_envs, action_dim) on CPU.
		"""
		obs  = obs.to(self.device)
		task = task.to(self.device) if task is not None else None

		zs = self.model.encode(obs, task)              # (B, latent_dim)
		action, _ = self.model.pi(zs, task)            # (B, action_dim)  [tanh applied]

		if task is not None:
			action = action * self.model._action_masks[task]

		if not eval_mode:
			noise  = torch.randn_like(action) * self.cfg.mrq_exploration_noise
			action = (action + noise).clamp(-1.0, 1.0)
			if task is not None:
				action = action * self.model._action_masks[task]

		return action.cpu()

	def update(self, buffer):
		"""
		One gradient step.  Samples from the mmbench buffer and runs the
		three MR.Q sub-updates (encoder → value → policy).

		Returns a dict of scalar training metrics.
		"""
		obs, action, reward, task = buffer.sample(device=self.device)
		# obs:    (H+1, B, obs_dim)   H = cfg.horizon
		# action: (H,   B, action_dim)
		# reward: (H,   B, 1)
		# task:   (H,   B) int32  or  None

		self.model.train()

		# ---- 1. Encoder auxiliary loss -----------------------------------
		enc_loss, enc_info = self._encoder_loss(obs, action, reward, task)

		self.encoder_optim.zero_grad(set_to_none=True)
		enc_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model.encoder_params(), self.cfg.grad_clip_norm)
		self.encoder_optim.step()

		# ---- 2. Value (TD3) loss -----------------------------------------
		value_loss, value_info = self._value_loss(obs, action, reward, task)

		self.value_optim.zero_grad(set_to_none=True)
		value_loss.backward()
		torch.nn.utils.clip_grad_norm_(
			self.model.value_params(), self.cfg.grad_clip_norm)
		self.value_optim.step()

		# ---- 3. Policy loss ----------------------------------------------
		# Freeze Q parameters BEFORE the forward so no Q gradients accumulate.
		for p in self.model.value_params():
			p.requires_grad_(False)
		policy_loss, policy_info = self._policy_loss(obs, task)
		self.policy_optim.zero_grad(set_to_none=True)
		policy_loss.backward()
		self.policy_optim.step()
		for p in self.model.value_params():
			p.requires_grad_(True)

		# ---- 4. Soft target update --------------------------------------
		self.model.soft_update_targets()

		self.model.eval()
		return {**enc_info, **value_info, **policy_info}

	# ------------------------------------------------------------------
	# Sub-update helpers
	# ------------------------------------------------------------------

	def _encoder_loss(self, obs, action, reward, task):
		"""
		MRQ auxiliary encoder loss.  Trains {_encoder, _za, _zsa, _dynamics_model}.

		Unrolls the dynamics model for H = cfg.horizon steps:
		  for t in 0..H-1:
		      zsa_t              = _zsa( zs_t ‖ _za(a_t) )
		      pred_next_zs, r_t  = _dynamics_model( zsa_t )
		      dyn_loss_t         = MSE( pred_next_zs, encode_target(obs[t+1]) )
		      reward_loss_t      = soft_ce( r_t, reward[t] )   [two-hot targets]
		      zs_{t+1}           = pred_next_zs                [model-based roll-out]

		The target encoder (EMA copy) provides stable supervision targets;
		no gradients flow through it.  Gradients accumulate across all H steps.

		Total loss = Σ_t  mrq_dyn_weight * dyn_loss_t
		                 + mrq_reward_weight * reward_loss_t
		"""
		# Use mrq_enc_horizon steps; cap to what the buffer actually returned.
		H = min(getattr(self.cfg, 'mrq_enc_horizon', obs.shape[0] - 1), obs.shape[0] - 1)
		task_0 = task[0] if task is not None else None

		# Initial real encoding
		zs = self.model.encode(obs[0], task_0)  # (B, latent_dim)

		# Pre-compute target encodings (no grad)
		with torch.no_grad():
			target_zs = [
				self.model.encode_target(obs[t + 1], task[t] if task is not None else None)
				for t in range(H)
			]

		enc_loss    = torch.tensor(0.0, device=self.device)
		total_dyn   = torch.tensor(0.0, device=self.device)
		total_rew   = torch.tensor(0.0, device=self.device)

		for t in range(H):
			pred_next_zs, pred_reward = self.model.predict(zs, action[t])

			dyn_loss    = F.mse_loss(pred_next_zs, target_zs[t].detach())
			reward_loss = math.soft_ce(
				pred_reward,
				reward[t],
				self.cfg,
			).mean()

			enc_loss  = enc_loss  + self.cfg.mrq_dyn_weight    * dyn_loss
			enc_loss  = enc_loss  + self.cfg.mrq_reward_weight * reward_loss
			total_dyn = total_dyn + dyn_loss
			total_rew = total_rew + reward_loss

			zs = pred_next_zs  # model-based unrolling

		info = {
			'enc_loss':    enc_loss.item() / H,
			'dyn_loss':    total_dyn.item() / H,
			'reward_loss': total_rew.item() / H,
		}
		return enc_loss, info

	def _value_loss(self, obs, action, reward, task):
		"""
		Bellman loss with H-step returns.  Trains all mrq_num_q Q-networks.

		Target (all inside torch.no_grad):
		  next_zs   = encode_target( obs[H], task )
		  ã         = clip( π_target(next_zs) + ε,  ε ~ N(0, σ_target) )
		  Q_tgt     = min over randomly-sampled 2 target Qs
		  disc      = γ[task]   per-task discount from cfg.discounts
		  R_H       = Σ_{t=0}^{H-1} disc^t * r_t  +  disc^H * Q_tgt

		Online Q (gradients flow through all _Qs; encoder detached):
		  zs        = encode( obs[0], task )          ← inside no_grad
		  zsa       = encode_zsa( zs, action[0] )     ← inside no_grad
		  loss      = avg over all Q-networks of smooth_l1(Q_i(zsa), R_H)
		"""
		H = obs.shape[0] - 1
		task_0 = task[0] if task is not None else None
		task_H = task[-1] if task is not None else None

		with torch.no_grad():
			# Target encoding of final obs
			next_zs = self.model.encode_target(obs[-1], task_H)   # (B, latent_dim)

			# TD3 target action with clipped noise
			next_action = self.model.pi_target(next_zs, task_H)
			noise = (torch.randn_like(next_action) * self.cfg.mrq_target_policy_noise
					 ).clamp(-self.cfg.mrq_noise_clip, self.cfg.mrq_noise_clip)
			next_action = (next_action + noise).clamp(-1.0, 1.0)
			if task is not None:
				next_action = next_action * self.model._action_masks[task_H]

			# Target Q
			next_zsa = self.model.encode_zsa_target(next_zs, next_action)
			Q_tgt    = self.model.Q_target(next_zsa)   # (B, 1)

			# Multi-step discounted return (no terminations in mmbench)
			# discount: (B,) per-task discount factors looked up by task ID
			if task is not None:
				disc = self.discount[task_0].view(-1, 1)  # (B, 1)
			else:
				disc = torch.full((reward[0].shape[0], 1),
								  self.discount[0].item(), device=self.device)
			ms_reward     = torch.zeros_like(reward[0])   # (B, 1)
			disc_accum    = torch.ones_like(ms_reward)
			for t in range(H):
				ms_reward  = ms_reward + disc_accum * reward[t]
				disc_accum = disc_accum * disc

			if getattr(self.cfg, 'mrq_reward_norm', 'none') == 'mrq':
				# Update reward scale: running EMA of mean absolute reward (matches
				# original MR.Q buffer.reward_scale() = abs().mean()).
				new_scale = reward.abs().mean().clamp(min=1e-8)
				self._target_reward_scale.copy_(self._reward_scale)
				self._reward_scale.lerp_(new_scale.view(1), self.cfg.tau)
				Q_target = (ms_reward + disc_accum * Q_tgt * self._target_reward_scale) / self._reward_scale
			else:
				Q_target = ms_reward + disc_accum * Q_tgt  # (B, 1)

			# Online encoding of initial obs (detached → no encoder grad)
			zs  = self.model.encode(obs[0], task_0)
			zsa = self.model.encode_zsa(zs, action[0])     # (B, latent_dim)

		# Online Q: train ALL Q-networks against the same TD target.
		num_q = len(self.model._Qs)
		if getattr(self.cfg, 'mrq_distributional_q', False):
			# Distributional: soft-CE loss on raw logits vs scalar TD target.
			total_loss = torch.tensor(0.0, device=self.device)
			q_scalars = []
			for q_net in self.model._Qs:
				q_logits = q_net(zsa)  # (B, num_bins)
				total_loss = total_loss + math.soft_ce(q_logits, Q_target, self.cfg).mean()
				q_scalars.append(math.two_hot_inv(q_logits, self.cfg))
			value_loss = total_loss / num_q
			Q = torch.cat(q_scalars, dim=1)  # (B, num_q) for logging
		else:
			# Scalar smooth-L1: one loss per Q-network, averaged.
			all_qs = torch.cat([q_net(zsa) for q_net in self.model._Qs], dim=1)  # (B, num_q)
			value_loss = F.smooth_l1_loss(all_qs, Q_target.expand(-1, num_q))
			Q = all_qs

		info = {
			'value_loss': value_loss.item(),
			'Q_mean':     Q.mean().item(),
			'Q_target':   Q_target.mean().item(),
		}
		return value_loss, info

	def _policy_loss(self, obs, task):
		"""
		TD3 max-Q policy loss with pre-activation regularisation.  Trains {_pi}.

		Gradient flow:
		  zs        = encode( obs[0], task )    ← inside no_grad (encoder frozen)
		  a, pre    = _pi( zs ‖ task_emb )      ← grads flow through _pi
		  zsa       = encode_zsa( zs, a )       ← grads flow through _za, _zsa, and a
		                                            but NOT through zs (already detached)
		  Q_pi      = Q_0(zsa)  if num_q==2 (TD3 original)           ← _Qs frozen
		             avg(Q_i, Q_j) for random i,j  if num_q>2 (NEWT-style)
		  loss      = -mean(Q_pi) + λ * mean(pre²)

		Note: _za and _zsa accumulate gradients here but policy_optim only
		updates _pi; the stale grads on _za/_zsa are cleared by
		encoder_optim.zero_grad(set_to_none=True) at the next update step.
		"""
		task_0 = task[0] if task is not None else None

		with torch.no_grad():
			zs = self.model.encode(obs[0], task_0)   # detached

		pi_action, pre_activ = self.model.pi(zs, task_0)
		if task is not None:
			pi_action = pi_action * self.model._action_masks[task_0]

		zsa_pi = self.model.encode_zsa(zs, pi_action)

		num_q = len(self.model._Qs)
		if num_q == 2:
			# Original MRQ behaviour: use first Q-network only (TD3 convention).
			Q_pi = self.model._Qs[0](zsa_pi)
			if getattr(self.cfg, 'mrq_distributional_q', False):
				Q_pi = math.two_hot_inv(Q_pi, self.cfg)
		else:
			# Ensemble (num_q > 2): randomly subsample 2, take average (NEWT-style).
			qidx = torch.randperm(num_q, device=self.device)[:2]
			q_vals = []
			for i in qidx:
				q_out = self.model._Qs[i](zsa_pi)
				if getattr(self.cfg, 'mrq_distributional_q', False):
					q_out = math.two_hot_inv(q_out, self.cfg)
				q_vals.append(q_out)
			Q_pi = (q_vals[0] + q_vals[1]) / 2  # (B, 1)

		if getattr(self.cfg, 'mrq_reward_norm', 'none') == 'newt':
			# Normalise Q by running IQR scale (mirrors TDMPC2 RunningScale).
			self._q_scale.update(Q_pi.detach())
			Q_pi = self._q_scale(Q_pi)

		policy_loss = (
			-Q_pi.mean()
			+ self.cfg.mrq_pre_activ_weight * pre_activ.pow(2).mean()
		)

		info = {
			'policy_loss':  policy_loss.item(),
			'pre_activ_sq': pre_activ.pow(2).mean().item(),
		}
		return policy_loss, info

	# ------------------------------------------------------------------
	# Checkpoint
	# ------------------------------------------------------------------

	def save(self, fp):
		ckpt = {
			'model':         self.model.state_dict(),
			'encoder_optim': self.encoder_optim.state_dict(),
			'value_optim':   self.value_optim.state_dict(),
			'policy_optim':  self.policy_optim.state_dict(),
		}
		norm = getattr(self.cfg, 'mrq_reward_norm', 'none')
		if norm == 'newt':
			ckpt['q_scale'] = self._q_scale.state_dict()
		elif norm == 'mrq':
			ckpt['reward_scale']        = self._reward_scale
			ckpt['target_reward_scale'] = self._target_reward_scale
		torch.save(ckpt, fp)

	def load(self, fp):
		sd = torch.load(fp, map_location=self.device, weights_only=False)
		self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
		norm = getattr(self.cfg, 'mrq_reward_norm', 'none')
		if norm == 'newt' and 'q_scale' in sd:
			self._q_scale.load_state_dict(sd['q_scale'])
		elif norm == 'mrq' and 'reward_scale' in sd:
			self._reward_scale.copy_(sd['reward_scale'])
			self._target_reward_scale.copy_(sd['target_reward_scale'])
