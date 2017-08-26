# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import numpy as np

Rsz=128 #replay buffer size
N=32
tau=0.9
gamma=0.9

critic=keras.Sequential()
actor=keras.Sequential()
criticp=critic.copy()
actorp=actor.copy()

env = gym.make('Pendulum-v0')

#replay buffers
Robs=np.zeros([Rsz] + list(env.observation_space.shape))
Robs1=np.zeros_like(Robs)
Rreward=np.zeros((Rsz,))
Raction=np.zeros([Rsz] + list(env.action_space.shape))

def exploration(action):
    return np.zeros_like(action)

ridx=0
Rfull=False

for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        action = actor.predict(observation)
        action += exploration(action)
        observation1, reward, done, info = env.step(action)
        Robs[ridx], Raction[ridx], Rreward[ridx], Robs1[ridx]=(observation, action, reward, observation1)
        env.render()

        ridx= (ridx + 1) % Rsz

        if Rfull:
            sample=np.random.choice(Rsz, N)
            y=Rreward[sample]+gamma*(criticp.predict([Robs[sample],actorp.predict(Robs1[sample])]))

            #update the critic
            critic.fit([Robs[sample],Raction[sample]],y,batch_size=N, epochs=1, verbose=1)

            #update the actor
            #critic.grad()*actor.grad()

            #update target networks
            for layerp,layer in zip(criticp.layers,critic.layers):
                layerp.set_weights(tau*layer.get_weights()+(1-tau)*layerp.get_weights())
            for layerp, layer in zip(actorp.layers, actor.layers):
                layerp.set_weights(tau * layer.get_weights() + (1 - tau) * layerp.get_weights())

        elif (ridx==0):
            Rfull=True

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        observation=observation1