import torch.nn as nn


def weight_init(m):
	"""Custom weight initialization for Newt."""
	if isinstance(m, nn.Linear):
		nn.init.trunc_normal_(m.weight, std=0.02)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)
	elif isinstance(m, nn.Parameter):
		if m.dim() == 3:
			nn.init.trunc_normal_(m, std=0.02)


def zero_(params):
	"""Initialize parameters to zero."""
	for p in params:
		p.data.fill_(0)
