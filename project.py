import gym
import keras
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from util import DDPGof,OrnsteinUhlenbeckProcess
import numpy as np

#tested environments
envname='Pendulum-v0'
#envname='MountainCarContinuous-v0'
import nservoarm
#envname='NServoArm-v0'


env = gym.make(envname)

#critic
oin = Input(shape=env.observation_space.shape,name='observeration')
ain = Input(shape=env.action_space.shape,name='action')
x=keras.layers.concatenate([oin, ain])
#x=BatchNormalization()(x)
#x=Dense(64, activation='relu')(x)
#x=Dense(64, activation='relu')(x)
#x=Dense(64, activation='relu')(x)
x=Dense(32, activation='relu')(x)
x=Dense(32, activation='relu')(x)
x=Dense(32, activation='relu')(x)
x=Dense(1, activation='linear', name='Q')(x)
critic=Model([oin,ain], x)
critic.compile(optimizer=Adam(lr=0.001),loss='mse')

#actor
x=oin
#x=BatchNormalization()(x)
x=Dense(32,input_shape=env.observation_space.shape)(x)
#x=Dense(32,activation='relu')(x)
#x=Dense(32,activation='relu')(x)
#x=Dense(32,activation='relu')(x)
x=Dense(16,activation='relu')(x)
x=Dense(16,activation='relu')(x)
x=Dense(16,activation='relu')(x)
x=Dense(env.action_space.shape[0],activation='linear')(x)
actor=Model(oin,x)
actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.0001), loss='mse')

nfrac=0.01 # exploration noise fraction
#todo: add support for unique sigma per dimension
exploration=OrnsteinUhlenbeckProcess(theta=.5, mu=0., dt=1.0,size=env.action_space.shape,
                             sigma=nfrac*np.max(env.action_space.high),sigma_min=nfrac*np.min(env.action_space.low))

# Observation normalization
observationOffset= (env.observation_space.low + env.observation_space.high) / 2
observationScale= 1 / (env.observation_space.high - env.observation_space.low)