from env.intersection import Intersection
from env.TSC_env import TSCEnv


class BaseAgent:
    def __init__(self, config, env: TSCEnv, idx):  # One agent corresponds to one intersection
        self.config = config
        self.env = env  # type: TSCEnv
        self.idx = idx
        self.cur_agent = self.config[self.config['cur_agent']]  # type: dict

        self.inter = env.n_intersection[idx]  # type: Intersection
        self.obs_shape = env.n_obs_shape[idx]
        action_space = env.n_action_space[idx]
        self.num_phase = action_space.n

        self.current_phase = 0
        self.device = config['device']

    def reset(self):
        raise NotImplementedError

    def pick_action(self, n_obs, on_training):
        raise NotImplementedError

    def learn(self):
        pass

    def store_experience(self, obs, action, reward, next_obs, done):
        pass
