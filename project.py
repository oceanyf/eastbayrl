import gym
from gym.spaces import Tuple
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Conv2D,MaxPool2D,Flatten,Lambda,Activation
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K
from util import DDPGof,ornstein_exploration
import numpy as np


datadir="../data"
modeldir="../data/models/{}"
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
        max_episode_steps=100,
        kwargs={'ngoals': 1,'image_goal':(160,320)}
    )
    env=gym.make('NServoArm-v0')
    return env

env = make_arm()

# returns x,y coordinates[0-1) of maximum value for each channel
def expected_pos(x):
    s1 = K.sum(x,axis=-2)
    s2 = K.sum(s1,axis=-2)
    s2 = 1.0/s2
    xc = K.variable(np.arange(int(x.shape[1]))/(int(x.shape[1])-1))
    x1 = K.variable(np.ones([x.shape[1]]))
    yc = K.variable(np.arange(int(x.shape[2]))/(int(x.shape[2])-1))
    y1 = K.variable(np.ones([x.shape[2]]))
    xx = K.dot(yc, x)
    xx = K.dot(x1, xx)
    xy = K.dot(y1, x)
    xy = K.dot(xc, xy)
    nc=K.stack([xx,xy],axis=-1)
    nc=K.transpose(K.transpose(nc)*K.transpose(s2))
    return nc

#create actor,critic
def make_models(locator=False):
    #critic

    ain = Input(shape=env.action_space.shape,name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')
    if hasattr(env.env,'image_goal'):
        x = Lambda(lambda x: x[:, 2:],name='image_only')(oin)
        iin=keras.layers.Reshape((env.env.height,env.env.width,3))(x)
        x=iin
        #x=BatchNormalization()(x) # image part
        x=Conv2D(16,(3,3),activation='relu')(x)
        #x=MaxPool2D((2,2),strides=(2,2))(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        #x=Dropout(.5)(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        #x=MaxPool2D((2,2),strides=(2,2))(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        #x=Dropout(.5)(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        #x=Dropout(.5)(x)
        x=Conv2D(16,(3,3),activation='relu')(x)
        x=Conv2D(4,(3,3),activation='softmax',name='image_softmax')(x)
        #x=Activation(keras.activations.softmax,axis=-2)(x)
        #x=Activation(keras.activations.softmax,axis=-3)(x)
        x = Lambda(expected_pos,name="expected_feature_location")(x)
        flat=Flatten(name='flattend_feature')(x)
        x = Dense(64, activation='relu')(flat)
        x = Dense(2, activation='linear')(x)
        locatormodel=Model(oin,x)
        locatormodel.compile(optimizer=Adam(lr=0.001), loss='mse')
        sensors = Lambda(lambda x: x[:,:2],name='sensors_only')(oin)
        cin = keras.layers.concatenate([sensors,flat],name='sensor_image')
    else:
        cin = keras.layers.concatenate([oin, ain],name='sensor')

    x=keras.layers.concatenate([cin, ain], name='sensor_action')
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
    critic.compile(optimizer=Adam(lr=0.1),loss='mse')

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
    actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.001), loss='mse')
    if locator:
        return actor, critic,locatormodel
    else:
        return actor,critic

actor,critic=make_models()

#exploration policy
exploration=ornstein_exploration(env.action_space,theta=.5, mu=0., dt=1.0,)
