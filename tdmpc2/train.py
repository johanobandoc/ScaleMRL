import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn
import hydra
from hydra.core.config_store import ConfigStore
from termcolor import colored
from tensordict import TensorDict

from common import barrier, set_seed
from common.buffer import Buffer, EnsembleBuffer
from common.logger import Logger
from common.world_model import WorldModel
from config import Config, split_by_rank, parse_cfg
from envs import make_env
from tdmpc2 import TDMPC2
from mrq import MRQModel, MRQ
from trainer import Trainer

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

cs = ConfigStore.instance()
cs.store(name="config", node=Config)


def setup(rank, world_size, port):
	os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
	os.environ["MASTER_PORT"] = port
	torch.distributed.init_process_group(
		backend="nccl",
		rank=rank,
		world_size=world_size
	)
	return port


class DDPWrapper(nn.Module):
	def __init__(self, module: nn.Module):
		super().__init__()
		self._module = module  # Can be plain or DDP-wrapped

	def forward(self, *args, **kwargs):
		return self._module(*args, **kwargs)

	def __getattr__(self, name):
		if name == '_module':
			return super().__getattr__(name)
		try:
			return getattr(self._module, name)
		except AttributeError:
			# Try to unwrap once if wrapped by DDP
			if hasattr(self._module, 'module'):
				return getattr(self._module.module, name)
			raise

	def __setattr__(self, name, value):
		if name == '_module':
			super().__setattr__(name, value)
		else:
			setattr(self._module, name, value)

	def state_dict(self, *args, **kwargs):
		return self._module.state_dict(*args, **kwargs)

	def load_state_dict(self, *args, **kwargs):
		return self._module.load_state_dict(*args, **kwargs)


def train(rank: int, cfg: dict, buffer):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.
	See config.yaml for a full list of args.
	"""
	# Set rank/device first so the buffer lands on the correct GPU
	cfg.rank = rank
	torch.cuda.set_device(rank)
	if cfg.world_size > 1:
		setup(rank, cfg.world_size, cfg.port)
		print(colored('Rank:', 'yellow', attrs=['bold']), rank)
	set_seed(cfg.seed + rank)

	# Multi-GPU: buffer is passed as an args dict to avoid pickling an
	# uninitialized LazyTensorStorage. Create it here in each process.
	if isinstance(buffer, dict):
		buffer_args = dict(buffer)
		buffer_args['multiproc'] = False  # each process owns its own buffer
		buffer_args['device'] = f"cuda:{rank}"  # each rank owns its buffer on its own GPU
		if cfg.use_demos:
			demo_buffer_args = deepcopy(buffer_args)
			demo_buffer_args['capacity'] = 1_900_000 if cfg.multitask else 50_000
			demo_buffer_args['batch_size'] = demo_buffer_args['batch_size'] // 2
			demo_buffer_args['cache_values'] = True
			buffer = EnsembleBuffer(Buffer(**demo_buffer_args), **buffer_args)
			load_demos(cfg, [buffer._offline, buffer])
		else:
			buffer = Buffer(**buffer_args)

	# split tasks across processes by rank
	if cfg.multitask:
		assert cfg.num_tasks % cfg.world_size == 0, \
			'Number of tasks must be divisible by number of GPUs.'
		cfg.tasks = split_by_rank(cfg.tasks, rank, cfg.world_size)
		print(f'[Rank {rank}] Tasks: {cfg.tasks}')
		cfg.num_tasks = len(cfg.tasks)
		cfg.num_envs = len(cfg.tasks)

	def make_agent(cfg):
		if cfg.agent == 'mrq':
			model = MRQModel(cfg).to(f"cuda:{cfg.rank}")
			if cfg.world_size > 1:
				model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.rank])
				model = DDPWrapper(model)
			return MRQ(model, cfg)
		model = WorldModel(cfg).to(f"cuda:{cfg.rank}")
		if cfg.world_size > 1:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.rank])
			model = DDPWrapper(model)
		agent = TDMPC2(model, cfg)
		return agent

	trainer = Trainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=make_agent(cfg),
		buffer=buffer,
		logger=Logger(cfg),
	)
	barrier()  # Ensure all processes are ready before starting training
	try:
		trainer.train()
		if cfg.rank == 0:
			print('\nTraining completed successfully')
	except Exception as e:
		print(colored(f'[Rank {cfg.rank}] Training crashed with exception: {repr(e)}', 'red', attrs=['bold']))
		raise  # Propagate the exception so it isn't swallowed
	except KeyboardInterrupt:
		print(colored(f'[Rank {cfg.rank}] Training interrupted by user (Ctrl+C)', 'red', attrs=['bold']))
		raise  # Optional: raise again if you want the shell to show KeyboardInterrupt
	finally:
		print(colored('Training interrupted', 'red', attrs=['bold']))
		if torch.distributed.is_initialized():
			torch.distributed.destroy_process_group()


def load_demos(
		cfg: dict,
		buffers: List[Buffer] = [],
		expected_obs_dim: int = 128,
		expected_action_dim: int = 16,
	):
	"""
	Load demonstrations into the buffer.
	"""
	tds = []
	num_eps = 0
	tasks = cfg.global_tasks if cfg.multitask else [cfg.task]
	for i, task in enumerate(tasks):
		demo_path = f'{cfg.data_dir}/{task}.pt'
		if not os.path.exists(demo_path):
			print(f'No demonstrations found for task {task}, skipping.')
			continue
		td = torch.load(demo_path, weights_only=False)
		
		# Load image observations if specified
		if cfg.obs == 'rgb':
			if 'feat' not in td:
				print(f'Warning: no visual features found in demonstrations for task {task}, skipping.')
				continue
			td['obs'] = TensorDict({'state': td['obs'], 'rgb': td['feat']})  # Non-stacked features
		try:
			del td['feat']
			del td['feat-stacked']
		except:
			pass
		td['task'] = torch.full_like(td['reward'], i, dtype=torch.int32)
		num_new_eps = td['episode'].max().item() + 1
		td['episode'] = td['episode'] + num_eps
		if task.startswith('ms-'): # Limit to 20 episodes for maniskill3 tasks
			td = td[td['episode'] < num_eps + 20]
			num_new_eps = 20
		num_eps += num_new_eps
		tds.append(td)
		print(f'Loaded {num_new_eps} episodes for task {task}')
	assert len(tds) > 0, 'No demonstrations found for any task.'
	tds = torch.cat(tds, dim=0)
	for buffer in buffers:
		buffer.load_demos(tds)


@hydra.main(version_base=None, config_name="config")
def launch(cfg: Config):
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# Set batch size
	cfg.world_size = torch.cuda.device_count() if cfg.multiproc else 1
	if cfg.world_size > 1:
		print(colored(f'Using {cfg.world_size} GPUs', 'green', attrs=['bold']))
		assert cfg.batch_size % cfg.world_size == 0, \
			'Batch size must be divisible by number of GPUs.'
		print(colored('Effective batch size:', 'yellow', attrs=['bold']), cfg.batch_size)
		cfg.batch_size = cfg.batch_size // cfg.world_size
		print(colored('Per-GPU batch size:', 'yellow', attrs=['bold']), cfg.batch_size)

	# Create buffer args
	buffer_args = {
		'capacity': cfg.buffer_size,
		'batch_size': cfg.batch_size,
		'horizon': cfg.horizon,
		'multiproc': cfg.multiproc,
		'compile': cfg.compile,
	}

	if cfg.world_size > 1:
		# Pass buffer_args dict to each process; each spawned process creates
		# its own buffer to avoid pickling an uninitialized LazyTensorStorage.
		cfg.port = os.getenv("MASTER_PORT", str(12355 + int(os.getpid()) % 1000))
		torch.multiprocessing.spawn(
			train,
			args=(cfg, buffer_args),
			nprocs=cfg.world_size,
			join=True,
		)
	else:
		if cfg.use_demos:
			# Create demonstration buffer
			demo_buffer_args = deepcopy(buffer_args)
			demo_buffer_args['capacity'] = 1_900_000 if cfg.multitask else 50_000
			demo_buffer_args['batch_size'] = demo_buffer_args['batch_size'] // 2
			demo_buffer_args['cache_values'] = True
			buffer = EnsembleBuffer(Buffer(**demo_buffer_args), **buffer_args)
			load_demos(cfg, [buffer._offline, buffer])
		else:
			# Default to regular buffer
			buffer = Buffer(**buffer_args)
		train(0, cfg, buffer)


if __name__ == '__main__':
	launch()
