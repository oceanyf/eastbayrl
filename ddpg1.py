# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import numpy as np

N=32
tau=0.9

critic=keras.Sequential()
actor=keras.Sequential()
criticp=critic.copy()
actorp=actor.copy()

R=[]

def exploration(action):
    return np.zeros_like(action)

env = gym.make('Pendulum-v0')
for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        action = actor.predict(observation)
        action += exploration(action)
        observation1, reward, done, info = env.step(action)
        env.render()
        R.append((observation,action,reward,observation1))
        batch=np.random.choice(R,N)

        y=batch.r+tau*(criticp.predict(observation1,actorp.predict(observation1)))

        #update the critic
        #update the actor
        #update target networks

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        observation=observation1