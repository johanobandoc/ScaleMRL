from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimNorm(nn.Module):
	"""
	Simplicial normalization.
	Adapted from https://arxiv.org/abs/2204.00616.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.dim = cfg.simnorm_dim

	def forward(self, x):
		shp = x.shape
		x = x.view(*shp[:-1], -1, self.dim)
		x = F.softmax(x, dim=-1)
		return x.view(*shp)

	def __repr__(self):
		return f"SimNorm(dim={self.dim})"
	

class NormedLinear(nn.Linear):
	"""
	Linear layer with LayerNorm, activation.
	"""

	def __init__(self, *args, act=None, **kwargs):
		super().__init__(*args, **kwargs)
		self.ln = nn.LayerNorm(self.out_features)
		if act is None:
			act = nn.Mish(inplace=False)
		self.act = act

	def forward(self, x):
		x = super().forward(x)
		return self.act(self.ln(x))

	def __repr__(self):
		if isinstance(self.act, nn.Sequential):
			act = '[' + ', '.join([m.__class__.__name__ for m in self.act]) + ']'
		else:
			act = self.act.__class__.__name__
		return f"NormedLinear(in_features={self.in_features}, "\
			f"out_features={self.out_features}, "\
			f"bias={self.bias is not None}, "\
			f"act={act})"


def mlp(in_dim, mlp_dims, out_dim, act=None):
	"""
	Basic building block of TD-MPC2.
	MLP with LayerNorm, Mish activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(NormedLinear(dims[i], dims[i+1]))
	mlp.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


def policy(in_dim, mlp_dims, out_dim, act=None):
	"""
	Policy network for TD-MPC2.
	Vanilla MLP with ReLU activations.
	"""
	if isinstance(mlp_dims, int):
		mlp_dims = [mlp_dims]
	dims = [in_dim] + mlp_dims + [out_dim]
	mlp = nn.ModuleList()
	for i in range(len(dims) - 2):
		mlp.append(nn.Linear(dims[i], dims[i+1]))
		mlp.append(nn.ReLU())
	mlp.append(nn.Linear(dims[-2], dims[-1]))
	return nn.Sequential(*mlp)


class QEnsemble(nn.Module):
	"""
	Vectorized ensemble of Q-networks. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		in_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
		mlp_dims = 2*[cfg.mlp_dim]
		out_dim = max(cfg.num_bins, 1)
		self._Qs = nn.ModuleList([mlp(in_dim, mlp_dims, out_dim) for _ in range(cfg.num_q)])
		if cfg.compile:
			if cfg.rank == 0:
				print('Compiling QEnsemble forward...')
			self._forward = torch.compile(self._forward_impl, mode='reduce-overhead')
		else:
			self._forward = self._forward_impl
	
	def _forward_impl(self, x):
		outs = [q(x) for q in self._Qs]
		return torch.stack(outs, dim=0)

	def forward(self, x):
		return self._forward(x)


class QOnlineTargetEnsemble(nn.Module):
	"""
	Online and target Q-ensembles for TD-MPC2. DDP compatible.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.online = QEnsemble(cfg)
		self.target = deepcopy(self.online)
		self.tau = cfg.tau
		self.target.train(False)
		self.track_grad(False, network='target')

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		self.online.train(mode)
		self.target.train(False)
		return self
	
	def track_grad(self, mode=True, network='online'):
		"""
		Enables/disables gradient tracking of Q-networks.
		Avoids unnecessary computation during policy optimization.
		"""
		assert network in {'online', 'target'}
		module = self.online if network == 'online' else self.target
		for p in module.parameters():
			p.requires_grad_(mode)

	@torch.no_grad()
	def hard_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.copy_(op.data)

	@torch.no_grad()
	def soft_update_target(self):
		for tp, op in zip(self.target.parameters(), self.online.parameters()):
			tp.data.lerp_(op.data, self.tau)

	def forward(self, x, target=False):
		if target:
			return self.target(x)
		else:
			return self.online(x)
		

def enc(cfg, out={}):
	"""
	Returns a dictionary of encoders for each observation in the dict.
	"""
	if cfg.obs == 'state':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim, max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	elif cfg.obs == 'rgb':
		out['state'] = mlp(cfg.obs_shape['state'][0] + cfg.task_dim + cfg.obs_shape['rgb'][0], max(cfg.num_enc_layers-1, 1)*[cfg.enc_dim], cfg.latent_dim, act=SimNorm(cfg))
	else:
		raise NotImplementedError(f"Unexpected observation type: {cfg.obs}")
	return nn.ModuleDict(out)


def api_model_conversion(target_state_dict, source_state_dict):
	"""
	Attempts to automatically convert a model checkpoint (e.g. add/remove DDP 'module.' prefixes).
	"""
	encoder_key = 'module._encoder.state.0.weight'
	if encoder_key in source_state_dict and encoder_key not in target_state_dict:
		# Remove 'module.' prefix from all keys in source_state_dict
		source_state_dict = {k[len('module.'):]: v for k, v in source_state_dict.items()}
	if encoder_key in target_state_dict and encoder_key not in source_state_dict:
		# Add 'module.' prefix to all keys in source_state_dict
		source_state_dict = {'module.' + k: v for k, v in source_state_dict.items()}

	for key in ['_encoder.state.0.weight', 'module._encoder.state.0.weight']:
		if key in target_state_dict and key in source_state_dict and \
				target_state_dict[key].shape != source_state_dict[key].shape:
			# possible rgb input in target but not in source, we should pad
			print('Warning: unexpected shape mismatch in encoder weights, attempting to pad source weights...')
			pad = target_state_dict[key].shape[1] - source_state_dict[key].shape[1]
			assert pad > 0, 'pad should be positive'
			pad_tensor = torch.zeros(source_state_dict[key].shape[0], pad, device=source_state_dict[key].device)
			source_state_dict[key] = torch.cat([source_state_dict[key], pad_tensor], dim=1)

	if '_action_masks' in target_state_dict and '_action_masks' in source_state_dict and \
			source_state_dict['_action_masks'].shape != target_state_dict['_action_masks'].shape:
		# repeat first dimension to match
		source_state_dict['_action_masks'] = source_state_dict['_action_masks'].repeat(
			target_state_dict['_action_masks'].shape[0] // source_state_dict['_action_masks'].shape[0], 1)
		if '_task_emb.weight' in source_state_dict:
			source_state_dict['_task_emb.weight'] = source_state_dict['_task_emb.weight'].repeat(
				target_state_dict['_action_masks'].shape[0] // source_state_dict['_task_emb.weight'].shape[0], 1)
		
	if '_task_emb.weight' in source_state_dict and not '_task_emb.weight' in target_state_dict:
		# delete task embedding from source state dict
		source_state_dict.pop('_task_emb.weight', None)

	return source_state_dict


def print_mismatched_tensors(target_state_dict, source_state_dict):
	target_keys = set(target_state_dict.keys())
	source_keys = set(source_state_dict.keys())

	# Keys in source but not in target
	for key in source_keys - target_keys:
		print(f"[Extra in source] {key}: shape={tuple(source_state_dict[key].shape)}")

	# Keys in target but not in source
	for key in target_keys - source_keys:
		print(f"[Missing in source] {key}: expected shape={tuple(target_state_dict[key].shape)}")

	# Keys present in both but with shape mismatch
	for key in target_keys & source_keys:
		try:
			t_shape = tuple(target_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in target_state_dict] {key}: {e}")
			continue
		try:
			s_shape = tuple(source_state_dict[key].shape)
		except AttributeError as e:
			print(f"[Error accessing shape in source_state_dict] {key}: {e}")
			continue
		if t_shape != s_shape:
			print(f"[Shape mismatch] {key}: target={t_shape}, source={s_shape}")
