import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR


def soft_ce(pred, target, cfg):
	"""Computes the cross entropy loss between predictions and soft targets."""
	pred = F.log_softmax(pred, dim=-1)
	target = two_hot(target, cfg)
	return -(target * pred).sum(-1, keepdim=True)


def log_std(x, low, dif):
	return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps, log_std):
	"""Compute Gaussian log probability."""
	residual = -0.5 * eps.pow(2) - log_std
	log_prob = residual - 0.9189385175704956
	return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
	"""Apply squashing function."""
	mu = torch.tanh(mu)
	pi = torch.tanh(pi)
	squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
	log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
	return mu, pi, log_pi


def discount_heuristic(cfg, episode_length):
	"""
	Returns discount factor for a given episode length.
	Simple heuristic that scales discount linearly with episode length.
	Default values should work well for most tasks, but can be changed as needed.

	Args:
		episode_length (int): Length of the episode. Assumes episodes are of fixed length.

	Returns:
		float: Discount factor for the task.
	"""
	frac = episode_length/cfg.discount_denom
	return min(max((frac-1)/(frac), cfg.discount_min), cfg.discount_max)


def int_to_one_hot(x, num_classes):
	"""
	Converts an integer tensor to a one-hot tensor.
	Supports batched inputs.
	"""
	one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
	one_hot.scatter_(-1, x.unsqueeze(-1), 1)
	return one_hot


def symlog(x):
	"""
	Symmetric logarithmic function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * torch.log(1 + torch.abs(x))


def symexp(x):
	"""
	Symmetric exponential function.
	Adapted from https://github.com/danijar/dreamerv3.
	"""
	return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg):
	"""Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symlog(x)
	x = torch.clamp(symlog(x), cfg.vmin, cfg.vmax).squeeze(1)
	bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size)
	bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
	soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device, dtype=x.dtype)
	bin_idx = bin_idx.long()
	soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
	soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
	return soft_two_hot


def two_hot_inv(x, cfg):
	"""Converts a batch of soft two-hot encoded vectors to scalars."""
	if cfg.num_bins == 0:
		return x
	elif cfg.num_bins == 1:
		return symexp(x)
	dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device, dtype=x.dtype)
	x = F.softmax(x, dim=-1)
	x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
	return symexp(x)


def gumbel_softmax_sample(p, temperature=1.0, dim=1):
	"""Sample indices from a Gumbel-Softmax distribution."""
	logits = torch.log(p + 1e-9)
	gumbels = -torch.empty_like(logits).exponential_().log()
	y = (logits + gumbels) / temperature
	return y.argmax(dim=dim)


def masked_bc_per_timestep(pi_action, action, task, action_masks):
	"""
	pi_action, action: (T, B, A)  # here T = T_actions (i.e., 3)
	task: (1, B) int64
	action_masks: (num_tasks, A) in {0,1}
	returns: (T, B) = mean MSE over valid dims per sample, per timestep
	"""
	T, B, A = action.shape

	mask_ba = action_masks.index_select(0, task[0]).to(dtype=action.dtype, device=action.device)  # (B, A)
	se = F.mse_loss(pi_action, action, reduction='none')  # (T, B, A)

	num = (se * mask_ba.unsqueeze(0)).sum(dim=2)         # (T, B)
	den = mask_ba.sum(dim=1).clamp_min(1).unsqueeze(0)   # (1, B), ≥1 per your assumption
	return num / den                                     # (T, B)


def interp_dist(base_mean, base_std, pi_mean, pi_std, step, start, end):
	"""
	Linear interpolation between two Gaussian distributions
	N(base_mean, base_std) and N(pi_mean, pi_std).
	"""
	if step < start:
		w = 1.0
	else:
		num = max(0, step - start)
		den = max(1, end)
		w = max(1.0 - (num / den), 0)
	w = torch.as_tensor(w, device=base_mean.device, dtype=base_mean.dtype).view(1, 1, 1)

	# Linearly annealed mix of policy and base prior
	mean = w * pi_mean + (1.0 - w) * base_mean
	std = w * pi_std + (1.0 - w) * base_std

	return mean, std


class MultiWarmupConstantLR:
	"""
	Linear warmup to each param group's base LR, then hold constant.
	Works for one or many optimizers with per-group base LRs.
	Call .step(step) at the *start* of each iteration to avoid off-by-one.
	"""
	def __init__(self, optimizers, warmup_steps: int):
		self.optimizers = list(optimizers)
		self.warmup_steps = int(warmup_steps)
		self._step = 0
		self.schedulers = []

		# Define factor schedule: step=0 -> 1/warmup, ..., step=warmup-1 -> warmup/warmup=1.0
		def lr_lambda(step):
			if self.warmup_steps <= 0:
				return 1.0
			return min(1.0, float(step + 1) / float(self.warmup_steps))

		for opt in self.optimizers:
			# Store per-group target base LR and start from 0.0 so first update is small
			for pg in opt.param_groups:
				# base LR is what's currently set on the param group
				base = pg.get("lr", 0.0)
				pg.setdefault("initial_lr", base)  # used by PyTorch schedulers internally
				pg["lr"] = 0.0                      # start warmup from 0

			# LambdaLR multiplies each group's *base_lr* (taken from initial_lr if present)
			self.schedulers.append(LambdaLR(opt, lr_lambda, last_epoch=-1))
			
	def current_lr(self, opt_idx: int = 0, group_idx: int = 0):
		return self.optimizers[opt_idx].param_groups[group_idx]["lr"]
	
	def current_lrs(self):
		return [[pg["lr"] for pg in opt.param_groups] for opt in self.optimizers]

	def step(self, step: int | None = None):
		"""Step all schedulers. Call at the *start* of your iteration."""
		if step is None:
			for sched in self.schedulers:
				sched.step()       # advances internal step by +1
			self._step += 1
		else:
			for sched in self.schedulers:
				sched.step(step)   # set explicit global step
			self._step = int(step) + 1

	def state_dict(self):
		return {
			"step": self._step,
			"warmup_steps": self.warmup_steps,
			"schedulers": [s.state_dict() for s in self.schedulers],
		}

	def load_state_dict(self, state):
		self._step = state["step"]
		self.warmup_steps = state["warmup_steps"]
		for s, sd in zip(self.schedulers, state["schedulers"]):
			s.load_state_dict(sd)
