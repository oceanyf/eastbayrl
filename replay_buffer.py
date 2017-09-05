import numpy as np


class ReplayBuffer(object):
    def __init__(self, length=200000, obs_space_shape=[], action_space_shape=[]):
        """Return a Customer object whose name is *name* and starting
        balance is *balance*."""
        self.length = length
        self.obs_space_shape = obs_space_shape
        self.action_space_shape = action_space_shape
        self.obs = np.zeros([length] + list(obs_space_shape))
        self.obs1 = np.zeros([length] + list(obs_space_shape))
        self.rewards = np.zeros((length,))
        self.action = np.zeros([length] + list(action_space_shape))
        self.done = np.zeros((length,))
        self.index = 0
        self.full = False

    def reset(self):
        self.obs = np.zeros([self.length] + list(self.obs_space_shape))
        self.obs1 = np.zeros([self.length] + list(self.obs_space_shape))
        self.rewards = np.zeros((self.length,))
        self.action = np.zeros([self.length] + list(self.action_space_shape))
        self.done = np.zeros((self.length,))
        self.index = 0
        self.full = False

    def insert(self, obs, obs1, action, rewards, done):
        if self.index == self.length:
            self.index = 0
        self.obs[self.index] = obs
        self.obs1[self.index] = obs1
        self.action[self.index] = action
        self.rewards[self.index] = rewards
        self.done[self.index] = done
        self.index += 1
        if self.index == self.length:
            self.full = True

    def sample(self, batch_size):
        if self.full:
            sample = np.random.choice(self.length, batch_size) #use random sample for now
        else:
            sample = np.random.choice(self.index, batch_size)
        return sample

