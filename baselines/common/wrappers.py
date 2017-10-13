import gym

class PrintEpisodeReward(gym.Wrapper):
    def _reset(self):
        self.r = 0
        return self.env.reset()

    def _step(self, action):
        obs, r, done, info = self.env.step(action)
        self.r += r
        if done:
            print('[PrintEpisodeReward] {}'.format(self.r))
        return obs, r, done, info
