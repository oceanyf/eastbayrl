import gym
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout
from keras.models import Model
from keras.optimizers import Adam
from util import DDPGof,ornstein_exploration
import numpy as np

#tested environments
def make_pendulum():
    return gym.make('Pendulum-v0')
def make_car():
    return gym.make('MountainCarContinuous-v0')
def make_arm():
    import nservoarm
    return gym.make('NServoArm-v0')

renderFlag=True
env = make_arm()


#create actor,critic
#note: if one network has batchnorm, then the other must also
def make_models():
    #critic
    oin = Input(shape=env.observation_space.shape,name='observeration')
    ain = Input(shape=env.action_space.shape,name='action')
    x=keras.layers.concatenate([oin, ain])
    x=BatchNormalization()(x)
    x=Dense(64, activation='relu')(x)
    #x=Dropout(.5)(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    #x=Dropout(.5)(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(1, activation='linear', name='Q')(x)
    critic=Model([oin,ain], x)
    critic.compile(optimizer=Adam(lr=0.001),loss='mse')

    #actor
    x=oin
    x=BatchNormalization()(x)
    x=Dense(32,input_shape=env.observation_space.shape)(x)
    x=Dense(32,activation='relu')(x)
    x=Dense(32,activation='relu')(x)
    x=Dense(32,activation='relu')(x)
    x=Dense(16,activation='relu')(x)
    #x=Dropout(.5)(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(env.action_space.shape[0],activation='linear')(x)
    actor=Model(oin,x)
    actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.000005), loss='mse')
    return actor,critic

actor,critic=make_models()

#exploration policy
exploration=ornstein_exploration(env.action_space,theta=.5, mu=0., dt=1.0,)


# Observation normalization
observationOffset= (env.observation_space.low + env.observation_space.high) / 2
observationScale= 1 / (env.observation_space.high - env.observation_space.low)