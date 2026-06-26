from collections import deque
import numpy as np
import gymnasium as gym
import torch
from tensordict import TensorDict

from common.vision_encoder import PretrainedEncoder


class EmbeddingWrapper(gym.Wrapper):
	"""
	A wrapper that applies a pretrained vision encoder to the observations.
	Supports per-environment frame stacking for multi-task vectorized RGB inputs.
	"""

	def __init__(self, env, num_frames=1):
		super().__init__(env)
		self.encoder = PretrainedEncoder()
		self.num_envs = env.num_envs
		self.num_frames = num_frames
		self.observation_space['rgb'] = gym.spaces.Box(
			low=-float('inf'), high=float('inf'), shape=(num_frames * 768,), dtype=np.float32)
		self._frames = [deque(maxlen=num_frames) for _ in range(self.num_envs)]

	def _get_stacked_obs(self, encoded_obs, env_idx):
		"""Return stacked features for a single env index."""
		self._frames[env_idx].append(encoded_obs)
		while len(self._frames[env_idx]) < self.num_frames:
			self._frames[env_idx].append(encoded_obs)  # pad if needed
		return torch.cat(list(self._frames[env_idx]), dim=-1)

	def _stack_all(self, encoded_obs_batch):
		"""Stack all environments' frame stacks into a batch."""
		stacked = []
		for i in range(self.num_envs):
			stacked.append(self._get_stacked_obs(encoded_obs_batch[i], i))
		return torch.stack(stacked)

	def encode(self, obs):
		"""Assumes obs is a batch of shape [B, 3, 224, 224] uint8."""
		return self.encoder(obs)  # returns [B, 768]

	def reset(self, **kwargs):
		obs, info = self.env.reset(**kwargs)
		encoded = self.encode(obs['rgb'])  # [B, 768]
		for i in range(self.num_envs):
			self._frames[i].clear()
			for _ in range(self.num_frames):
				self._frames[i].append(encoded[i])
		obs['rgb'] = self._stack_all(encoded)
		return TensorDict(obs, batch_size=self.cfg.num_envs), info

	def _stack_encoded(self, env_idx, encoded):
		"""Helper to stack an already-encoded final observation."""
		self._frames[env_idx].clear()
		for _ in range(self.num_frames):
			self._frames[env_idx].append(encoded)
		return torch.cat(list(self._frames[env_idx]), dim=-1)

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		encoded = self.encode(obs['rgb'])  # [B, 768]

		done = terminated | truncated  # [B]
		done_indices = torch.nonzero(done).squeeze(-1)  # [D]

		final_obs = info.get('final_observation', None)
		if final_obs is not None and len(done_indices) > 0:
			if isinstance(final_obs, TensorDict):  # shape: [D, 3, 224, 224]
				final_obs_batch = final_obs['rgb']
			# Handle both List[Tensor] and Tensor
			elif isinstance(final_obs, torch.Tensor):
				final_obs_batch = final_obs  # shape: [D, 3, 224, 224]
			else:
				final_obs_batch = torch.stack(final_obs, dim=0)

			encoded_final = self.encode(final_obs_batch)  # [D, 768]

			# Return stacked embeddings for only the done envs
			final_obs_encoded = torch.stack([
				self._stack_encoded(env_idx.item(), encoded_final[j])
				for j, env_idx in enumerate(done_indices)
			], dim=0)  # [D, k×768]
			if isinstance(final_obs, TensorDict):
				info['final_observation']['rgb'] = final_obs_encoded
			else:
				info['final_observation'] = final_obs_encoded

		# Update frame buffers for all envs in a vectorized loop
		stacked_obs = []
		for i in range(self.num_envs):
			if done[i]:
				self._frames[i].clear()
				for _ in range(self.num_frames):
					self._frames[i].append(encoded[i])
			else:
				self._frames[i].append(encoded[i])
			stacked_obs.append(torch.cat(list(self._frames[i]), dim=-1))
		stacked_obs = torch.stack(stacked_obs)

		obs['rgb'] = stacked_obs

		return TensorDict(obs, batch_size=self.cfg.num_envs), reward, terminated, truncated, info
