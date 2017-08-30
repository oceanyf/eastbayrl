import gym
import keras
from keras.layers import Dense,Input,BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from util import DDPGof,OrnsteinUhlenbeckProcess
import nservoarm


env = gym.make('NServoArm-v0')

#critic
oin = Input(shape=env.observation_space.shape,name='observeration')
ain = Input(shape=env.action_space.shape,name='action')
#c1=Dense(2,activation='linear')(oin)
#c2=Dense(2,activation='linear')(ain)
c3=keras.layers.concatenate([oin,ain])
#c3=BatchNormalization()(c3)
c3=Dense(32,activation='relu')(c3)
c3=Dense(32,activation='relu')(c3)
c3=Dense(32,activation='relu')(c3)
c3=Dense(1,activation='linear',name='Q')(c3)
critic=Model([oin,ain],c3)
critic.compile(optimizer=Adam(lr=0.001),loss='mse')

#actor
actor=keras.models.Sequential()
actor.add(Dense(10,input_shape=env.observation_space.shape))
actor.add(Dense(16,activation='relu'))
actor.add(Dense(16,activation='relu'))
actor.add(Dense(16,activation='relu'))
actor.add(Dense(env.action_space.shape[0],activation='linear'))
actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.001), loss='mse')

exploration=OrnsteinUhlenbeckProcess(size=env.action_space.shape, sigma=2.14, theta=.15, mu=0.,sigma_min=0.3)