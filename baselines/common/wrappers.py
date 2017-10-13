import gym
import time
import json

class PrintEpisodeReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.t_start = None

    def _reset(self):
        if self.t_start is None:
            self.t_start = time.time()
        self.r = 0
        self.len = 0
        return self.env.reset()

    def _step(self, action):
        obs, r, done, info = self.env.step(action)
        self.r += r
        self.len += 1
        if done:
            data = {
                'time': time.time() - self.t_start,
                'ep_r': self.r,
                'ep_len': self.len,
            }
            print('[PrintEpisodeReward] ' + json.dumps(data))
        return obs, r, done, info
