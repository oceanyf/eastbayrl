import gym
from gym.spaces import Tuple
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Conv2D,MaxPool2D,Flatten,Lambda
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
    gym.envs.register(
        id='NServoArm-v0',
        entry_point='nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={'ngoals': 1,'image_goal':(160,320)}
    )
    env=gym.make('NServoArm-v0')
    print(env.env.observation_space)
    return env

env = make_arm()


#create actor,critic
def make_models():
    #critic
    print(env.observation_space.shape)
    ain = Input(shape=env.action_space.shape,name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')
    if hasattr(env.env,'image_goal'):
        #image_in = Input(shape=env.observation_space.spaces[1].shape,name='image_observation')
        x = Lambda(lambda x: x[:, 2:])(oin)
        x=keras.layers.Reshape((env.env.height,env.env.width,3))(x)
        x=BatchNormalization()(x) # image part
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=MaxPool2D((2,2),strides=(2,2))(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=MaxPool2D((2,2),strides=(2,2))(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=MaxPool2D((2,2),strides=(2,2))(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Flatten(name='flattendImage')(x)
        x=Dense(16, activation='relu',name='image_features')(x)
        sensors = Lambda(lambda x: x[:,:2])(oin)
        cin = keras.layers.concatenate([sensors,x],name='sensor_image')
    else:
        cin = keras.layers.concatenate([oin, ain],name='sensor')

    keras.layers.concatenate([cin, ain], name='sensor_action')
    x=Dense(64, activation='relu')(x)
    #x=Dropout(.5)(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu',kernel_regularizer='l2')(x)
    #x=Dropout(.5)(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu',kernel_regularizer='l2')(x)
    x=Dense(1, activation='linear', name='Q')(x)
    critic=Model([oin,ain], x)
    critic.compile(optimizer=Adam(lr=0.001),loss='mse')

    #actor
    x=cin
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(16,activation='relu',kernel_regularizer='l2')(x)
    #x=Dropout(.5)(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(env.action_space.shape[0],activation='linear')(x)
    actor=Model(oin, x)
    actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.0001), loss='mse')
    return actor,critic

actor,critic=make_models()

#exploration policy
exploration=ornstein_exploration(env.action_space,theta=.5, mu=0., dt=1.0,)


# Sensor Observation normalization
if isinstance(env.observation_space, Tuple):
    observationOffset= (env.observation_space.spaces[0].low + env.observation_space.spaces[0].high) / 2
    observationScale= 1 / (env.observation_space.spaces[0].high - env.observation_space.spaces[0].low)
else:
    observationOffset= (env.observation_space.low + env.observation_space.high) / 2
    observationScale= 1 / (env.observation_space.high - env.observation_space.low)