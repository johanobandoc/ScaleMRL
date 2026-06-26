import torch
from functools import partial
from termcolor import colored
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler


def transform(td: TensorDict, horizon: int = 3):
	td = td.select("obs", "action", "reward", "task", strict=False)
	td = td.view(-1, horizon + 1).transpose(0, 1)
	obs = td.get("obs").contiguous()
	action = td.get("action")[1:].contiguous()
	reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
	task = td.get("task", None)
	if task is not None:
		task = task[1:].contiguous()

	return obs, action, reward, task


class Buffer:
	"""
	Replay buffer for Newt training. Based on torchrl.
	Uses CUDA memory if available, and CPU memory otherwise.
	"""

	def __init__(
		self,
		capacity: int = 1_000_000,
		batch_size: int = 1024,
		horizon: int = 3,
		multiproc: bool = False,
		cache_values: bool = False,
		compile: bool = True,
		device: str = "cuda:0",
	):
		self.set_storage_device(device)

		self._capacity = capacity
		self._batch_size = batch_size
		self._horizon = horizon
		self._sample_size = batch_size * (horizon + 1)
		self._multiproc = multiproc

		self._sampler = SliceSampler(
			num_slices=batch_size,
			end_key=None,
			traj_key="episode",
			truncated_key=None,
			strict_length=True,
			cache_values=cache_values,
			use_gpu=True,
			compile=not self._multiproc and compile,
		)
		self._storage = LazyTensorStorage(self._capacity, device=self._storage_device)
		self._buffer = ReplayBuffer(
			storage=self._storage,
			sampler=self._sampler,
			pin_memory=False,
			prefetch=None if self._multiproc else 8,
			batch_size=self._sample_size,
			shared=self._multiproc,
			transform=partial(transform, horizon=self._horizon),
		)
		self._num_eps = 0
		self._num_demos = 0

	@property
	def capacity(self):
		return self._capacity

	@property
	def num_eps(self):
		return self._num_eps

	def set_storage_device(self, device):
		"""Set the storage device for the buffer."""
		if isinstance(device, str):
			device = torch.device(device)
		if hasattr(self, "_storage_device") and self._storage_device == device:
			return
		elif hasattr(self, "_storage_device"):
			print(f"[{self.__class__.__name__}] Changing storage device from {self._storage_device} to {device}.")
		else:
			print(f"[{self.__class__.__name__}] Setting storage device to {device}.")
		self._storage_device = device

	def print_requirements(self, tds):
		"""Use the first episode to estimate storage requirements."""
		print(f"[{self.__class__.__name__}] Buffer capacity: {self._capacity:,}")
		bytes_per_step = sum(
			[
				(v.numel() * v.element_size() if not isinstance(v, TensorDict) else
				 sum([x.numel() * x.element_size() for x in v.values()]))
				for v in tds.values()
			]
		) / len(tds)
		total_bytes = bytes_per_step * self._capacity
		print(f"[{self.__class__.__name__}] Storage required: {total_bytes/1e9:.2f} GB")
		print(f"[{self.__class__.__name__}] Using {self._storage_device} memory for storage.")

	def save(self, path):
		"""Save the buffer storage to disk."""
		assert self._num_eps > 0, "Buffer is empty, nothing to save."
		torch.save(self._buffer.storage._storage, path)

	def load_demos(self, tds):
		"""Load a demonstration dataset into the buffer."""
		assert self._num_eps == 0, "Expected an empty buffer when loading demos!"
		self._num_demos = tds["episode"].max().item() + 1
		self.print_requirements(tds[tds["episode"] == 0])
		self._buffer.extend(tds)
		print(
			colored(
				f"Added {self._num_demos} demonstrations to {self.__class__.__name__}. "
				f"Capacity: {len(self._buffer)}/{self.capacity}.",
				"green",
				attrs=["bold"],
			)
		)
		return self._num_demos

	def next_episode_id(self, world_size=1, rank=0):
		"""Return the next episode ID to be used (unique across ranks)."""
		return self._num_demos + self._num_eps * world_size + rank

	def add(self, td, world_size=1, rank=0):
		"""Add an episode to the buffer."""
		num_new_eps = td.shape[0]
		assert num_new_eps == 1, "Expected a single episode to be added at a time. Use `load` for multiple episodes."
		if self._num_eps == 0 and rank == 0:
			self.print_requirements(td[0])
		td["episode"] = torch.full_like(td["reward"], self.next_episode_id(world_size, rank), dtype=torch.int64)
		for i in range(num_new_eps):
			self._buffer.extend(td[i])
		self._num_eps += num_new_eps
		return self._num_eps

	def sample(self, device: torch.device):
		"""
		Sample a batch of subsequences from the buffer.

		Transform already did:
		  - select obs/action/reward/task
		  - reshape to [H+1, B] (obs) and [H, B] (action/reward/task)
		  - contiguous() on all leaves

		Here we only hop device (pinned -> GPU) and unpack.
		"""
		obs, action, reward, task = self._buffer.sample()
		if obs.device != device:
			obs = obs.to(device, non_blocking=True)
			action = action.to(device, non_blocking=True)
			reward = reward.to(device, non_blocking=True)
			if task is not None:
				task = task.to(device, non_blocking=True)

		return obs, action, reward, task


class EnsembleBuffer(Buffer):
	"""
	Replay buffer for co-training on offline and online data.
	"""

	def __init__(
		self,
		offline_buffer: Buffer,
		*args,
		**kwargs
	):
		kwargs['batch_size'] = kwargs['batch_size'] // 2  # Use half the batch size for each buffer
		self._offline = offline_buffer
		super().__init__(*args, **kwargs)

	def set_storage_device(self, device):
		self._offline.set_storage_device(device)
		super().set_storage_device(device)

	def sample(self, device):
		"""Sample a batch of subsequences from the two buffers."""
		obs0, action0, reward0, task0 = self._offline.sample(device)
		try:
			obs1, action1, reward1, task1 = super().sample(device)
		except Exception as e:
			print('Failed to sample from online buffer!', e)
			raise
		
		# Lazy one-time allocation of output tensors
		if not hasattr(self, '_out_obs'):
			B = self._batch_size * 2
			assert obs0.shape[1] == obs1.shape[1] == self._batch_size
			self._out_obs = torch.empty((self._horizon + 1, B) + obs0.shape[2:], dtype=obs0.dtype, device=device)
			self._out_action = torch.empty((self._horizon, B) + action0.shape[2:], dtype=action0.dtype, device=device)
			self._out_reward = torch.empty((self._horizon, B) + reward0.shape[2:], dtype=reward0.dtype, device=device)
			if task0 is not None and task1 is not None:
				self._out_task = torch.empty((self._horizon, B) + task0.shape[2:], dtype=task0.dtype, device=device)
			else:
				self._out_task = None

		# Write into preallocated tensors
		self._out_obs[:, :self._batch_size] = obs0
		self._out_obs[:, self._batch_size:] = obs1
		self._out_action[:, :self._batch_size] = action0
		self._out_action[:, self._batch_size:] = action1
		self._out_reward[:, :self._batch_size] = reward0
		self._out_reward[:, self._batch_size:] = reward1
		if task0 is not None and task1 is not None:
			self._out_task[:, :self._batch_size] = task0
			self._out_task[:, self._batch_size:] = task1
		
		return self._out_obs, self._out_action, self._out_reward, self._out_task
