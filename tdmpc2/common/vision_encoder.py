import torch
from transformers import AutoImageProcessor, AutoModel


__RGB_PROCESSOR__ = None
__RGB_ENCODER__ = None


def pcount(m):
	count = sum(p.numel() for p in m.parameters())
	return f"{count:,}"


class PretrainedEncoder():

	@property
	def mean_std(self):
		return dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	def __init__(self):
		if __RGB_ENCODER__ is None:
			self._init_encoder()

	def _init_encoder(self):
		global __RGB_PROCESSOR__, __RGB_ENCODER__
		__RGB_PROCESSOR__ = AutoImageProcessor.from_pretrained(
			'facebook/dinov2-base',
			use_fast=True,
		)
		__RGB_ENCODER__ = AutoModel.from_pretrained(
			'facebook/dinov2-base').to('cuda:0')
		__RGB_ENCODER__.eval()
		print('PretrainedEncoder: using', __RGB_ENCODER__.name_or_path)
		print('PretrainedEncoder: model size is', pcount(__RGB_ENCODER__))

		# Test the encoder
		x = (torch.clamp(torch.randn(1, 3, 224, 224, device=0) * 0.5 + 0.5, 0, 1) * 255).to(torch.uint8)
		print('PretrainedEncoder: output shape is', self(x).shape)

	@torch.no_grad()
	def _normalize(self, x):
		mean = torch.tensor(self.mean_std['mean'], device=x.device).view(1, 3, 1, 1)
		std = torch.tensor(self.mean_std['std'], device=x.device).view(1, 3, 1, 1)
		return (x - mean) / std

	@torch.no_grad()
	def forward(self, x: torch.Tensor):
		# Input: uint8 tensor in [0, 255], shape [B, 3, 224, 224]
		assert x.dtype == torch.uint8 and x.ndim == 4 and x.shape[1] == 3 \
			   and x.shape[2:] == (224, 224), \
			f"Expected uint8 [B,3,224,224], got {x.dtype} {x.shape}"
		x = x.to(__RGB_ENCODER__.device).to(dtype=torch.float32) / 255.0  # → [0, 1] float
		x = self._normalize(x)  # ImageNet normalization
		outputs = __RGB_ENCODER__(pixel_values=x)
		return outputs.last_hidden_state[:, 0]  # CLS token

	@torch.no_grad()
	def __call__(self, *args, **kwargs):
		return self.forward(*args, **kwargs)
