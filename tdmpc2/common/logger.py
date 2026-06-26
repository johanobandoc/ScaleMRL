import os
import datetime
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from termcolor import colored

from common import TASK_SET


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_score", "S", "float"),
	("elapsed_time", "T", "time"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}


def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("envs", cfg.num_envs*cfg.world_size),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	if cfg.multitask:
		kvs[0] = ("tasks", cfg.num_global_tasks)
		kvs[1] = ("world size", cfg.world_size)
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(Path(cfg.work_dir) / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and self._wandb and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 1:
			frames = np.stack(self.frames[:-1])
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)


class Logger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self.rank = cfg.rank
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if self.rank > 0 or not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			if self.rank == 0:
				print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			else:
				print(colored(f"Logging disabled for rank {self.rank}.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = False
			self._save_agent = False
			self._wandb = None
			self._video = None
			return
		self._log_dir = Path(make_dir(cfg.work_dir))
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		print_run(cfg)
		import wandb
		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(cfg.seed),
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=cfg,
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = (
			VideoRecorder(cfg, self._wandb)
			if self._wandb and cfg.save_video
			else None
		)

	@property
	def video(self):
		return self._video

	def save_agent(self, agent=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)

	def finish(self, agent=None):
		if agent is not None:
			self.save_agent(agent)
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.03f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def pprint_multitask(self, d, cfg):
		"""Pretty-print evaluation metrics for multi-task training."""
		if self.rank > 0:
			return
		print(colored(f'Evaluated agent on {cfg.num_global_tasks} tasks:', 'yellow', attrs=['bold']))
		scores = defaultdict(list)
		domains = [k for k in TASK_SET.keys() if k != 'soup']
		for k, v in d.items():
			if '+' not in k:
				continue
			task = k.split('+')[1]
			if k.startswith('episode_score'):
				for domain in domains:
					if task in TASK_SET[domain]:
						scores[f'avg_score_{domain}'].append(v)
						print(colored(f'  {task:<34}\tS: {v:.03f}', 'yellow'))
						break
				scores['avg_score'].append(v)

		# Normalized score
		for domain, score in scores.items():
			scores[domain] = np.mean(score) if len(score) > 0 else float('nan')
	
		# Print summary
		for domain, score in scores.items():
			if domain.startswith('avg_score_'):
				print(colored(f'{domain[10:]:<34}\tS: {score:.03f}', 'yellow', attrs=['bold']))
		print(colored(f'{"unweighted score":<34}\tS: {scores["avg_score"]:.03f}', 'yellow', attrs=['bold']))
		scores['avg_score_weighted'] = np.nanmean([scores[domain] for domain in scores if domain.startswith('avg_score_')])
		print(colored(f'{"weighted score":<34}\tS: {scores["avg_score_weighted"]:.03f}', 'yellow', attrs=['bold']))
		d.update(scores)

	def pprint_pretrain(self, d):
		if self.rank > 0:
			return
		print(colored('-'*30 + '\nPretraining metrics:', 'yellow', attrs=['bold']))
		for k, v in d.items():
			print(colored(f' {k:<22}{v:.05f}', 'yellow'))
		print(colored('-'*30, 'yellow'))

	def log(self, d, category="train"):
		if self.rank > 0:
			return
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d["step"])
		if category in {'train', 'eval'}:
			self._print(d, category)
