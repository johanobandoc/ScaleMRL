import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import json
import hydra
from termcolor import colored
from omegaconf import OmegaConf

from common import MODEL_SIZE, TASK_SET
from common.math import discount_heuristic


@dataclass
class Config:
	"""
	Config for experiments.
	"""

	# environment
	task: str = "soup"										# "soup" for multitask, see tdmpc2/common/__init__.py for task list
	obs: str = "state"										# observation type, one of ["state", "rgb"]
	num_envs: int = 10										# number of parallel environments, overridden if task is "soup"
	env_mode: str = "async"									# environment mode, one of ["async", "sync"]

	# evaluation
	checkpoint: Optional[str] = None						# path to model checkpoint for evaluation / finetuning
	eval_episodes: int = 2									# number of evaluation episodes per parallel environment

	# training
	steps: int = 100_000_000								# total environment steps to train for
	batch_size: int = 1024									# effective batch size across all devices
	utd: float = 0.075										# update-to-data ratio, i.e., model updates per environment step
	reward_coef: float = 0.1								# coefficient for reward prediction loss
	value_coef: float = 0.1									# coefficient for value prediction loss
	consistency_coef: float = 20.0							# coefficient for latent consistency loss
	prior_coef: float = 10.0								# coefficient for bc prior loss
	rho: float = 0.5										# temporal weight coefficient for model losses
	lr: float = 3e-4										# global learning rate
	enc_lr_scale: float = 0.3								# encoder learning rate scale (wrt global lr)
	grad_clip_norm: float = 20.0							# gradient clipping norm
	tau: float = 0.01										# target value network update rate
	discount_denom: int = 5									# denominator for discount factor heuristic
	discount_min: float = 0.95								# minimum discount factor
	discount_max: float = 0.995								# maximum discount factor
	buffer_size: int = 10_000_000							# replay buffer capacity
	use_demos: bool = True									# whether to use demonstration data
	demo_steps: int = 200_000								# number of pretraining steps on demonstration data
	lr_schedule: Optional[str] = None						# learning rate schedule, one of [None, "warmup"]
	warmup_steps: int = 5_000								# number of warmup steps for lr schedule
	seeding_coef: int = 5									# number of random rollouts (per env) to seed the buffer with
	exp_name: str = "default"								# experiment name for logging
	finetune: bool = False									# enable when finetuning a multitask pretrained model on a single task

	# planning
	mpc: bool = True										# whether to use planning for action selection
	iterations: int = 6										# number of planning iterations
	num_samples: int = 512									# number of action sequences sampled per iteration
	num_elites: int = 64									# number of elite action sequences to refit distribution
	num_pi_trajs: int = 24									# number of action sequences to sample from policy prior
	horizon: int = 3										# planning horizon (also determines model training horizon)
	min_std: float = 0.05									# minimum action sampling stddev
	max_std: float = 2.0									# maximum action sampling stddev
	temperature: float = 0.5								# softmax temperature for mppi weighting
	constrained_planning: bool = True						# whether to constrain planning after pretraining
	constraint_start_step: int = 2_000_000					# you probably want this to be seeding_coef * num_envs * episode_length
	constraint_final_step: int = 10_000_000					# linearly anneal constraint weight until this step

	# actor
	log_std_min: float = -10								# min log stddev for actor
	log_std_max: float = 2.0								# max log stddev for actor
	entropy_coef: float = 1e-4								# coefficient for actor entropy bonus

	# critic
	num_bins: int = 101										# number of bins for discrete regression
	vmin: float = -10.0										# min (log) value for discrete regression
	vmax: float = +10.0										# max (log) value for discrete regression

	# architecture
	model_size: Optional[str] = None						# model size, see tdmpc2/common/__init__.py for options
	num_enc_layers: int = 3									# number of encoder layers, overridden by model_size
	enc_dim: int = 1024										# encoder mlp width, overridden by model_size
	mlp_dim: int = 1024										# model mlp width, overridden by model_size
	latent_dim: int = 512									# model latent state dim, overridden by model_size
	task_dim: int = 512										# task embedding dim, 512 assumes CLIP embeddings
	num_q: int = 5											# number of Q-functions in ensemble, overridden by model_size
	simnorm_dim: int = 8									# number of dims per simplex in simplicial embedding layer

	# logging
	wandb_project: str = "<project>"						# wandb project name
	wandb_entity: str = "<user>"							# wandb entity (user) name
	enable_wandb: bool = True								# whether to enable wandb logging

	# agent
	agent: str = "tdmpc2"									# agent type, one of ["tdmpc2", "mrq"]

	# mrq-specific hyperparameters (only used when agent="mrq")
	mrq_za_dim: int = 256									# MRQ action embedding dimension
	mrq_exploration_noise: float = 0.2						# TD3 exploration noise std
	mrq_target_policy_noise: float = 0.2					# TD3 target-policy smoothing noise std
	mrq_noise_clip: float = 0.3								# TD3 noise clipping bound
	mrq_pre_activ_weight: float = 1e-5						# policy pre-activation regularisation
	mrq_dyn_weight: float = 1.0								# encoder dynamics loss weight
	mrq_reward_weight: float = 0.1							# encoder reward loss weight
	mrq_reward_norm: str = "newt"							# reward/Q normalisation: "none" | "mrq" | "newt"
	mrq_enc_horizon: int = 5								# encoder unroll steps (original MRQ paper default); capped to cfg.horizon at runtime
	mrq_policy_layernorm: bool = True						# if True: policy hidden layers use LayerNorm+Mish (closer to original MRQ); if False: bare ReLU (current default)
	mrq_distributional_q: bool = True						# if True: Q outputs num_bins logits + soft-CE loss (like NEWT); if False: scalar Q + smooth-L1 (original MRQ default)
	mrq_num_q: int = 2										# number of Q-networks (original MRQ twin-Q default)
	mrq_num_q_from_model_size: bool = True					# if True: ignore mrq_num_q and use cfg.num_q (model-size-dependent, same as NEWT: S→3, B/L→5, XL→7)

	# misc
	multiproc: bool = False									# whether to use multiple GPUs (will use all visible GPUs)
	compile: bool = True									# whether to use torch.compile for model compilation (faster)
	render_size: int = 224									# render size for rgb observations
	save_video: bool = False								# whether to save evaluation videos
	save_agent: bool = True									# whether to save agent checkpoints
	data_dir: str = "<path>/<to>/data"						# directory for demonstrations
	seed: int = 1											# random seed

	# convenience (filled at runtime)
	work_dir: Optional[str] = None
	task_title: Optional[str] = None
	tasks: Any = None
	global_tasks: Any = None
	num_tasks: Optional[int] = None
	num_global_tasks: Optional[int] = None
	task_embeddings: Any = None
	obs_shape: Any = None
	action_dim: Optional[int] = None
	episode_length: Optional[int] = None
	obs_shapes: Any = None
	action_dims: Any = None
	episode_lengths: Any = None
	discounts: Any = None
	eval_freq: Optional[int] = None
	save_freq: Optional[int] = None
	bin_size: Optional[float] = None
	rank: int = 0
	world_size: int = 1
	port: Optional[str] = None
	child_env: bool = False
	multitask: bool = False

	get = lambda self, val, default=None: getattr(self, val, default)


def split_by_rank(global_list, rank, world_size):
	"""Split a global list into sublists for each rank."""
	return [global_list[i] for i in range(len(global_list)) if i % world_size == rank]


def parse_cfg(cfg):
	"""
	Parses the experiment config dataclass. Mostly for convenience.
	"""
	# Convenience
	cfg.work_dir = Path(hydra.utils.get_original_cwd()) / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1)  # Bin size for discrete regression

	# Model size
	if not cfg.multitask and cfg.get('model_size', None) is None:
		cfg.model_size = 'B'  # Default model size for single-task training (5M)
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v

	# Set defaults
	cfg.tasks = TASK_SET.get(cfg.task, [cfg.task] * cfg.num_envs)
	cfg.multitask = cfg.task in TASK_SET  # True for any named task set (soup, atari, dmcontrol, ...)
	cfg.num_tasks = len(dict.fromkeys(cfg.tasks))  # Unique tasks
	cfg.global_tasks = deepcopy(cfg.tasks)
	cfg.num_global_tasks = cfg.num_tasks
	if cfg.multitask:
		cfg.num_envs = cfg.num_tasks
		print(colored(f'Number of tasks: {cfg.num_global_tasks}', 'green', attrs=['bold']))
	else:
		cfg.use_demos = False  # Disable demos for single-task training
		cfg.task_dim = 0  # No task conditioning for single-task training
	cfg.eval_freq = 20 * 500 * cfg.num_envs
	cfg.save_freq = 5 * cfg.eval_freq

	# Warmup LR when pretraining
	if cfg.use_demos and cfg.checkpoint is None:
		cfg.lr_schedule = "warmup"

	# Check if save_video and env_mode are compatible
	if cfg.save_video:
		assert cfg.env_mode == "sync", "save_video is only compatible with env_mode 'sync'"

	# Load task info and embeddings
	curr_dir = os.path.dirname(os.path.abspath(__file__))
	tasks_fp = os.path.join(curr_dir, '..', 'tasks.json')
	assert os.path.exists(tasks_fp), f'Task info file not found at {tasks_fp}'
	with open(tasks_fp, "r") as f:
		task_info = json.load(f)
	cfg.task_embeddings = []
	cfg.episode_lengths = []
	cfg.discounts = []
	cfg.action_dims = []
	for task in cfg.tasks:
		assert task in task_info, f'Task {task} not found in task embeddings.'
		cfg.task_embeddings.append(task_info[task]['text_embedding'])
		cfg.episode_lengths.append(task_info[task]['max_episode_steps'])
		if 'discount_factor' in task_info[task]:
			cfg.discounts.append(task_info[task]['discount_factor'])
		else:
			cfg.discounts.append(discount_heuristic(cfg, task_info[task]['max_episode_steps']))
		cfg.action_dims.append(task_info[task]['action_dim'])

	return OmegaConf.to_object(cfg)
