import numpy as np
import gymnasium as gym

import envs.tasks.pygame as games


PYGAME_TASKS = {
	'pygame-cowboy': games.CowboyEnv,
	'pygame-coinrun': games.CoinRunEnv,
	'pygame-spaceship': games.SpaceshipEnv,
	'pygame-pong': games.PongEnv,
	'pygame-bird-attack': games.BirdAttackEnv,
	'pygame-highway': games.HighwayEnv,
	'pygame-landing': games.LandingEnv,
	'pygame-air-hockey': games.AirHockeyEnv,
	'pygame-rocket-collect': games.RocketCollectEnv,
	'pygame-chase-evade': games.ChaseEvadeEnv,
	'pygame-coconut-dodge': games.CoconutDodgeEnv,
	'pygame-cartpole-balance': games.CartpoleBalanceEnv,
	'pygame-cartpole-swingup': games.CartpoleSwingupEnv,
	'pygame-cartpole-balance-sparse': games.CartpoleBalanceSparseEnv,
	'pygame-cartpole-swingup-sparse': games.CartpoleSwingupSparseEnv,
	'pygame-cartpole-tremor': games.CartpoleTremorEnv,
	'pygame-point-maze-var1': games.PointMazeVariant1Env,
	'pygame-point-maze-var2': games.PointMazeVariant2Env,
	'pygame-point-maze-var3': games.PointMazeVariant3Env,
	# below are reserved for testing
	'pygame-point-maze-var4': games.PointMazeVariant4Env,
	'pygame-reacher-easy': games.ReacherEasyEnv,
	'pygame-reacher-hard': games.ReacherHardEnv,
}


class PygameWrapper(gym.Wrapper):
	def __init__(self, env, cfg):
		super().__init__(env)
		self.env = env
		self.cfg = cfg
		if cfg.obs == 'rgb':
			self.observation_space = gym.spaces.Dict({
				'rgb': gym.spaces.Box(
					low=0, high=255, shape=(3, self.cfg.render_size, self.cfg.render_size), dtype=np.uint8),
				'state': env.observation_space,
			})
		self._cumulative_reward = 0

	def _extract_info(self, info):
		info = {
			'terminated': info.get('terminated', False),
			'truncated': info.get('truncated', False),
			'success': float(info.get('success', 0.)),
		}
		if 'cowboy' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 500, 0, 1)
		elif 'coinrun' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 50, 0, 1)
		elif 'spaceship' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 60, 0, 1)
		elif 'pong' in self.cfg.task:  # convert [-3, 3] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 3) / 6, 0, 1)
		elif 'bird-attack' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 16, 0, 1)
		elif 'highway' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 10, 0, 1)
		elif 'landing' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 200, 0, 1)
		elif 'air-hockey' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 10, 0, 1)
		elif 'rocket-collect' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 25, 0, 1)
		elif 'chase-evade' in self.cfg.task:  # convert [-50, 50] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 50) / 100, 0, 1)
		elif 'coconut-dodge' in self.cfg.task:  # convert [-15, 0] to [0, 1]
			info['score'] = np.clip((self._cumulative_reward + 15) / 15, 0, 1)
		elif 'cartpole' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 500, 0, 1)
		elif 'point-maze' in self.cfg.task:
			info['score'] = info['success']
		elif 'reacher' in self.cfg.task:
			info['score'] = np.clip(self._cumulative_reward / 200, 0, 1)
		else:
			info['score'] = 0
		return info

	def get_observation(self, obs):
		if self.cfg.obs == 'rgb':
			return {'state': obs, 'rgb': self.render().transpose(2, 0, 1)}
		return obs

	def reset(self):
		obs, info = self.env.reset()
		self._cumulative_reward = 0
		return self.get_observation(obs), self._extract_info(info)

	def step(self, action):
		obs, reward, _, truncated, info = self.env.step(action.copy())
		terminated = False
		self._cumulative_reward += reward
		info['terminated'] = terminated
		info['truncated'] = truncated
		return self.get_observation(obs), reward, terminated, truncated, self._extract_info(info)

	@property
	def unwrapped(self):
		return self.env.unwrapped
	
	def render(self, **kwargs):
		return self.env.render()


def make_env(cfg):
	"""
	Make Pygame (MiniArcade) environment.
	"""
	if not cfg.task in PYGAME_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	env = PYGAME_TASKS[cfg.task]()
	env = PygameWrapper(env, cfg)
	return env
