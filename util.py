from __future__ import division
import numpy as np
import keras.backend as K

# Warning: If the critic uses BatchNormalization, then the actor must also.
def DDPGof(opt):
    class tmp(opt):
        def __init__(self, critic, actor, *args, **kwargs):
            super(tmp, self).__init__(*args, **kwargs)
            self.critic=critic
            self.actor=actor
        def get_gradients(self,loss, params):
            self.combinedloss= -self.critic([self.actor.inputs[0],self.actor.outputs[0]])
            return K.gradients(self.combinedloss,self.actor.trainable_weights)
    return tmp

from matplotlib.widgets import CheckButtons


class ToggleFlags:
    def __init__(self,kwargs=None):
        self.names=[]
        self.kwargs=kwargs
    def add(self,name,value=False):
        if name in ['add','showat','__init__']:return
        if name in self.kwargs:
            value=self.kwargs[name]
        self.names.append(name)
        self.__setattr__(name,value)
    def showat(self,ax):
        v=[self.__getattribute__(name) for name in self.names]
        self.check=CheckButtons(ax,self.names,v)
        def func(label):
            self.__setattr__(label,not self.__getattribute__(label))
            print("clicked")
        self.check.on_clicked(func)
# wrap environment and add UI


# borrowed from https://github.com/matthiasplappert/keras-rl.git

class RandomProcess(object):
    def reset_states(self):
        pass


class AnnealedGaussianProcess(RandomProcess):
    def __init__(self, mu, sigma, sigma_min, n_steps_annealing):
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma


class GaussianWhiteNoiseProcess(AnnealedGaussianProcess):
    def __init__(self, mu=0., sigma=1., sigma_min=None, n_steps_annealing=1000, size=1):
        super(GaussianWhiteNoiseProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.size = size

    def sample(self):
        sample = np.random.normal(self.mu, self.current_sigma, self.size)
        self.n_steps += 1
        return sample

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, theta, mu=0., sigma=1., dt=1e-2, x0=None, size=1, sigma_min=None, n_steps_annealing=1000):
        super(OrnsteinUhlenbeckProcess, self).__init__(mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.size = size
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)


def ornstein_exploration(space,theta,nfrac=0.01,**kwrds):
    #todo: add support for unique sigma per dimension
    return OrnsteinUhlenbeckProcess(theta,size=space.shape,
                                 sigma=nfrac*np.max(space.high),sigma_min=nfrac*np.min(space.low),**kwrds)