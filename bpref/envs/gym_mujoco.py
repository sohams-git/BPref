import gym
import numpy as np

class GymMujocoEnv:
    def __init__(self, env_id, add_time=False, max_ep_len=1000, seed=None):
        self.env = gym.make(env_id)

        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass

        self.env_id = env_id
        self.add_time = add_time
        self.max_ep_len = max_ep_len
        self._max_episode_steps = int(max_ep_len)
        self.t = 0

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        if add_time:
            low  = np.concatenate([self.observation_space.low,  [0.0]])
            high = np.concatenate([self.observation_space.high, [1.0]])
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def _augment(self, obs):
        if not self.add_time:
            return obs
        return np.concatenate([obs, [min(self.t / float(self.max_ep_len), 1.0)]], axis=-1)

    def reset(self):
        self.t = 0
        obs = self.env.reset()
        return self._augment(obs)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self.t += 1
        done = done or (self.t >= self.max_ep_len)
        return self._augment(obs), float(r), bool(done), info

    def seed(self, s):
        try:
            self.env.seed(s)
        except Exception:
            pass