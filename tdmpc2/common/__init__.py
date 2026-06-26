import random

import numpy as np
import torch


MODEL_SIZE = {
	'S':   {'enc_dim': 128,
		  	'mlp_dim': 256,
		  	'latent_dim': 384,
		  	'num_enc_layers': 2,
			'num_q': 3},
	'B':   {'enc_dim': 256,
		  	'mlp_dim': 512,
		  	'latent_dim': 512,
		  	'num_enc_layers': 2},
	'L':   {'enc_dim': 1024,
		  	'mlp_dim': 1024,
		  	'latent_dim': 512,
		  	'num_enc_layers': 3},
	'XL':  {'enc_dim': 2048,
			'mlp_dim': 2048,
			'latent_dim': 704,
			'num_enc_layers': 4,
			'num_q': 7},
}

TASK_SET = {
	'dmcontrol': [  # 21 tasks -> dmcontrol
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop', 'quadruped-walk',
		'quadruped-run',
	],
	'dmcontrol-ext': [  # 16 tasks -> dmcontrol-ext
		'walker-walk-backward', 'walker-run-backward', 'cheetah-run-backward', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backward', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin', 'jumper-jump', 'spinner-spin', 'spinner-spin-backward', 'spinner-jump',
		'giraffe-run',
	],
    'metaworld': [  # 49 tasks ->metaworld
		'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
		'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
		'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
		'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
		'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
		'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
		'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
		'mw-reach', 'mw-reach-wall', 'mw-soccer', 'mw-stick-push', 'mw-stick-pull',
		'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close', 'mw-bin-picking',
		'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
	],
    'maniskill': [  # 36 tasks -> maniskill
		'ms-ant-walk', 'ms-ant-run', 'ms-cartpole-balance', 'ms-cartpole-swingup', 'ms-hopper-stand',
		'ms-hopper-hop', 'ms-pick-cube', 'ms-pick-cube-eepose', 'ms-pick-cube-so', 'ms-poke-cube',
		'ms-push-cube', 'ms-pull-cube', 'ms-pull-cube-tool', 'ms-stack-cube', 'ms-place-sphere',
		'ms-lift-peg', 'ms-pick-apple', 'ms-pick-banana', 'ms-pick-can', 'ms-pick-hammer',
		'ms-pick-fork', 'ms-pick-knife', 'ms-pick-mug', 'ms-pick-orange', 'ms-pick-screwdriver',
		'ms-pick-spoon', 'ms-pick-tennis-ball', 'ms-pick-baseball', 'ms-pick-cube-xarm6', 'ms-pick-sponge',
		'ms-anymal-reach', 'ms-reach', 'ms-reach-eepose', 'ms-reach-xarm6', 'ms-cartpole-balance-sparse',
		'ms-cartpole-swingup-sparse',
	],
	'mujoco': [  # 6 tasks->mujoco
        'mujoco-ant', 'mujoco-halfcheetah', 'mujoco-hopper', 'mujoco-inverted-pendulum', 'mujoco-reacher',
		'mujoco-walker',
	],
	'box2d': [  # 8 tasks
        'bipedal-walker-flat', 'bipedal-walker-uneven', 'bipedal-walker-rugged', 'bipedal-walker-hills', 'bipedal-walker-obstacles',
		'lunarlander-land', 'lunarlander-hover', 'lunarlander-takeoff',
	],
	'robodesk': [  # 6 tasks
		'rd-push-red', 'rd-push-green', 'rd-push-blue', 'rd-open-slide', 'rd-open-drawer',
		'rd-flat-block-in-bin',
	],
	'ogbench': [  # 12 tasks
		'og-ant', 'og-antball', 'og-point-arena', 'og-point-maze', 'og-point-bottleneck',
		'og-point-circle', 'og-point-spiral', 'og-ant-arena', 'og-ant-maze', 'og-ant-bottleneck',
		'og-ant-circle', 'og-ant-spiral',
	],
	'pygame': [  # 19 tasks
		'pygame-cowboy', 'pygame-coinrun', 'pygame-spaceship', 'pygame-pong', 'pygame-bird-attack',
		'pygame-highway', 'pygame-landing', 'pygame-air-hockey', 'pygame-rocket-collect', 'pygame-chase-evade',
		'pygame-coconut-dodge', 'pygame-cartpole-balance', 'pygame-cartpole-swingup', 'pygame-cartpole-balance-sparse', 'pygame-cartpole-swingup-sparse',
		'pygame-cartpole-tremor', 'pygame-point-maze-var1', 'pygame-point-maze-var2', 'pygame-point-maze-var3',
	],
	'atari': [  # 27 tasks -> "atari"
		'atari-alien', 'atari-assault', 'atari-asterix', 'atari-atlantis', 'atari-bank-heist',
		'atari-battle-zone', 'atari-beamrider', 'atari-boxing', 'atari-chopper-command', 'atari-crazy-climber',
		'atari-double-dunk', 'atari-gopher', 'atari-ice-hockey', 'atari-jamesbond', 'atari-kangaroo',
		'atari-krull', 'atari-ms-pacman', 'atari-name-this-game', 'atari-phoenix', 'atari-pong',
		'atari-road-runner', 'atari-robotank', 'atari-seaquest', 'atari-space-invaders', 'atari-tutankham',
		'atari-upndown', 'atari-yars-revenge',
	],
	'soup': [  # 200 tasks in total
		# dmcontrol (21 tasks)
		'walker-stand', 'walker-walk', 'walker-run', 'cheetah-run', 'reacher-easy',
	    'reacher-hard', 'acrobot-swingup', 'pendulum-swingup', 'cartpole-balance', 'cartpole-balance-sparse',
		'cartpole-swingup', 'cartpole-swingup-sparse', 'cup-catch', 'finger-spin', 'finger-turn-easy',
		'finger-turn-hard', 'fish-swim', 'hopper-stand', 'hopper-hop', 'quadruped-walk',
		'quadruped-run',
		# dmcontrol-ext (16 tasks)
		'walker-walk-backward', 'walker-run-backward', 'cheetah-run-backward', 'cheetah-run-front', 'cheetah-run-back',
		'cheetah-jump', 'hopper-hop-backward', 'reacher-three-easy', 'reacher-three-hard', 'cup-spin',
		'pendulum-spin', 'jumper-jump', 'spinner-spin', 'spinner-spin-backward', 'spinner-jump',
		'giraffe-run',
		# meta-world (49 tasks)
		'mw-assembly', 'mw-basketball', 'mw-button-press-topdown', 'mw-button-press-topdown-wall', 'mw-button-press',
		'mw-button-press-wall', 'mw-coffee-button', 'mw-coffee-pull', 'mw-coffee-push', 'mw-dial-turn',
		'mw-disassemble', 'mw-door-open', 'mw-door-close', 'mw-drawer-close', 'mw-drawer-open',
		'mw-faucet-open', 'mw-faucet-close', 'mw-hammer', 'mw-handle-press-side', 'mw-handle-press',
		'mw-handle-pull-side', 'mw-handle-pull', 'mw-lever-pull', 'mw-peg-insert-side', 'mw-peg-unplug-side',
		'mw-pick-out-of-hole', 'mw-pick-place', 'mw-pick-place-wall', 'mw-plate-slide', 'mw-plate-slide-side',
		'mw-plate-slide-back', 'mw-plate-slide-back-side', 'mw-push-back', 'mw-push', 'mw-push-wall',
		'mw-reach', 'mw-reach-wall', 'mw-soccer', 'mw-stick-push', 'mw-stick-pull', 
		'mw-sweep-into', 'mw-sweep', 'mw-window-open', 'mw-window-close', 'mw-bin-picking',
		'mw-box-close', 'mw-door-lock', 'mw-door-unlock', 'mw-hand-insert',
		# maniskill (36 tasks)
		'ms-ant-walk', 'ms-ant-run', 'ms-cartpole-balance', 'ms-cartpole-swingup', 'ms-hopper-stand',
		'ms-hopper-hop', 'ms-pick-cube', 'ms-pick-cube-eepose', 'ms-pick-cube-so', 'ms-poke-cube',
		'ms-push-cube', 'ms-pull-cube', 'ms-pull-cube-tool', 'ms-stack-cube', 'ms-place-sphere',
		'ms-lift-peg', 'ms-pick-apple', 'ms-pick-banana', 'ms-pick-can', 'ms-pick-hammer',
		'ms-pick-fork', 'ms-pick-knife', 'ms-pick-mug', 'ms-pick-orange', 'ms-pick-screwdriver',
		'ms-pick-spoon', 'ms-pick-tennis-ball', 'ms-pick-baseball', 'ms-pick-cube-xarm6', 'ms-pick-sponge',
		'ms-anymal-reach', 'ms-reach', 'ms-reach-eepose', 'ms-reach-xarm6', 'ms-cartpole-balance-sparse',
		'ms-cartpole-swingup-sparse',
		# mujoco (6 tasks)
		'mujoco-ant', 'mujoco-halfcheetah', 'mujoco-hopper', 'mujoco-inverted-pendulum', 'mujoco-reacher',
		'mujoco-walker',
		# box2d (8 tasks)
		'bipedal-walker-flat', 'bipedal-walker-uneven', 'bipedal-walker-rugged', 'bipedal-walker-hills', 'bipedal-walker-obstacles',
		'lunarlander-land', 'lunarlander-hover', 'lunarlander-takeoff',
		# robodesk (6 tasks)
		'rd-push-red', 'rd-push-green', 'rd-push-blue', 'rd-open-slide', 'rd-open-drawer',
		'rd-flat-block-in-bin',
		# ogbench (12 tasks)
		'og-ant', 'og-antball', 'og-point-arena', 'og-point-maze', 'og-point-bottleneck',
		'og-point-circle', 'og-point-spiral', 'og-ant-arena', 'og-ant-maze', 'og-ant-bottleneck',
		'og-ant-circle', 'og-ant-spiral',
		# pygame (19 tasks)
		'pygame-cowboy', 'pygame-coinrun', 'pygame-spaceship', 'pygame-pong', 'pygame-bird-attack',
		'pygame-highway', 'pygame-landing', 'pygame-air-hockey', 'pygame-rocket-collect', 'pygame-chase-evade',
		'pygame-coconut-dodge', 'pygame-cartpole-balance', 'pygame-cartpole-swingup', 'pygame-cartpole-balance-sparse', 'pygame-cartpole-swingup-sparse',
		'pygame-cartpole-tremor', 'pygame-point-maze-var1', 'pygame-point-maze-var2', 'pygame-point-maze-var3',
		# atari (27 tasks)
		'atari-alien', 'atari-assault', 'atari-asterix', 'atari-atlantis', 'atari-bank-heist',
		'atari-battle-zone', 'atari-beamrider', 'atari-boxing', 'atari-chopper-command', 'atari-crazy-climber',
		'atari-double-dunk', 'atari-gopher', 'atari-ice-hockey', 'atari-jamesbond', 'atari-kangaroo',
		'atari-krull', 'atari-ms-pacman', 'atari-name-this-game', 'atari-phoenix', 'atari-pong',
		'atari-road-runner', 'atari-robotank', 'atari-seaquest', 'atari-space-invaders', 'atari-tutankham',
		'atari-upndown', 'atari-yars-revenge',
	],
}


def set_seed(seed):
	"""Set seed for reproducibility."""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def barrier():
	if torch.distributed.is_initialized():
		try:
			torch.distributed.barrier()
		except Exception:
			raise
