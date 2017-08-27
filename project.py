import gym
import keras
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model

env = gym.make('Pendulum-v0')

#critic
oin = Input(shape=env.observation_space.shape,name='observeration')
ain = Input(shape=env.action_space.shape,name='action')
#c1=Dense(2,activation='linear')(oin)
#c2=Dense(2,activation='linear')(ain)
c3=keras.layers.concatenate([oin,ain])
#c3=BatchNormalization()(c3)
c3=Dense(3)(c3)
c3=Dense(1,activation='linear',name='Q')(c3)
critic=Model([oin,ain],c3)
critic.compile(optimizer='adam',loss='mse')

#actor
actor=keras.models.Sequential()
actor.add(Dense(10,input_shape=env.observation_space.shape))
actor.add(Dense(1,activation='linear'))
actor.compile(optimizer='adam',loss='mse')