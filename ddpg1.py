# from scratch attempt at ddpg with split V+A
#
import gym
import keras
from keras.layers import Dense,Input,Concatenate
from keras.models import Model
import keras.backend as K
import numpy as np


Rsz=1000 #replay buffer size
N=32 #mini batch size
tau=0.001
gamma=0.99
std=0.01

#start the environment
env = gym.make('Pendulum-v0')
print(env.observation_space)
print(env.action_space)

#critic
oin = Input(shape=env.observation_space.shape,name='observeration')
ain = Input(shape=env.action_space.shape,name='action')
c1=Dense(10,activation='linear')(oin)
c2=Dense(10,activation='linear')(ain)
c3=keras.layers.concatenate([c1,c2])
c3=Dense(10)(c3)
c3=Dense(1,activation='linear',name='Q')(c3)
critic=Model([oin,ain],c3)
critic.summary()
critic.compile(optimizer='adam',loss='mse')

#actor
actor=keras.models.Sequential()
actor.add(Dense(10,input_shape=env.observation_space.shape))
actor.add(Dense(1,activation='linear'))
actor.summary()
actor.compile(optimizer='adam',loss='mse')

#target networks
criticp=keras.models.clone_model(critic)
criticp.compile(optimizer='adam',loss='mse')
actorp=keras.models.clone_model(actor)
actorp.compile(optimizer='adam',loss='mse')




#replay buffers
Robs=np.zeros([Rsz] + list(env.observation_space.shape))
Robs1=np.zeros_like(Robs)
Rreward=np.zeros((Rsz,))
Raction=np.zeros([Rsz] + list(env.action_space.shape))
Rdone=np.zeros((Rsz,))

def exploration(action):
    return np.random.normal(0.0,std,action.shape)

ridx=0
Rfull=False

for i_episode in range(2000):
    observation1 = env.reset()
    totalReward=0
    for t in range(1000):
        observation=observation1
        action = actor.predict(np.expand_dims(observation,axis=0))
        action += exploration(action)
        observation1, reward, done, _ = env.step(action)
        observation1=observation1[:,0]
        Robs[ridx]=observation
        Raction[ridx]=action
        Rreward[ridx]=reward
        Rdone[ridx]=done
        Robs1[ridx]=observation1
        ridx= (ridx + 1) % Rsz

        if Rfull and ridx%10 == 0:
            sample=np.random.choice(Rsz, N)

            #update critic
            yq=Rreward[sample]+gamma*(criticp.predict([Robs1[sample],actorp.predict(Robs1[sample])])[:,0])
            critic.fit([Robs[sample],Raction[sample]],yq,batch_size=N, epochs=1,verbose=0)

            #update the actor : critic.grad()*actor.grad()
            cgrad = K.gradients(critic.outputs, critic.inputs[1]) # grad of Q wrt actions
            f=K.function(critic.inputs,cgrad)
            actions=actor.predict(Robs[sample])
            grads=f([Robs[sample],actions])[0]
            ya=actions+0.1*grads  # nudge action in direction that improves Q
            actor.fit(Robs[sample], ya, verbose=0, epochs=1)
            #actor.fit(Robs[sample],grads,verbose=0,epochs=1) # I think this is what others have done.

            #update target networks
            criticp.set_weights([tau*w+(1-tau)*wp for wp,w in zip(criticp.get_weights(),critic.get_weights())])
            actorp.set_weights([tau*w+(1-tau)*wp for wp,w in zip(actorp.get_weights(),actor.get_weights())])
        elif (ridx==0):
            Rfull=True

        env.render()
        totalReward+=reward
        if done:
            print("Episode finished after {} timesteps total reward={}".format(t + 1, totalReward))
            break
