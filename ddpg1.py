# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
import os

print(dir(os))

Rsz=200000 #replay buffer size
N=32 # sample size
tau=0.01
gamma=0.95
Qscale=(1/(1-gamma))
Oscale=np.array([1,1,.2])
std=0.005
warmup=50

#import project specifics, such as actor/critic models
from project import *
print('observation space {} high {} low {}'.format(env.observation_space,env.observation_space.high,env.observation_space.low))
print('action space {} high {} low {}'.format(env.action_space,env.action_space.high,env.action_space.low))
critic.summary()
actor.summary()

# create target networks
criticp=keras.models.clone_model(critic)
criticp.compile(optimizer='adam',loss='mse')
criticp.set_weights(critic.get_weights())
actorp=keras.models.clone_model(actor)
actorp.compile(optimizer='adam',loss='mse')
actorp.set_weights(actor.get_weights())

# define gradient function for ddgp
cgrada=K.gradients(critic.outputs, critic.inputs[1])
cgradaf = K.function(critic.inputs,cgrada )  # grad of Q wrt actions


#allocate replay buffers
Robs=np.zeros([Rsz] + list(env.observation_space.shape))
Robs1=np.zeros_like(Robs)
Rreward=np.zeros((Rsz,))
Raction=np.zeros([Rsz] + list(env.action_space.shape))
Rdone=np.zeros((Rsz,))

#define exploration policy
def exploration(action):
    return np.random.normal(0.12,std,action.shape)



#set up the plotting
plt.ion()
plt.figure(1)
renderFlag=False
def ontype(event): # any keypress will toggle the rendering
    global renderFlag
    renderFlag=not renderFlag
plt.gcf().canvas.mpl_connect('key_press_event',ontype)


rcnt=0
Rfull=False
RewardsHistory = []
QAccHistory = []
for i_episode in range(200000):
    observation1 = env.reset()
    episodeReward=0
    clipcnt=0
    episode=[]
    for t in range(1000):
        observation=observation1

        #take step using the action based on actor
        action = actor.predict(np.expand_dims(observation,axis=0))
        action += exploration(action)
        observation1, reward, done, _ = env.step(action)
        observation1=observation1[:,0]*Oscale

        # insert into replay buffer
        ridx=rcnt%Rsz
        Robs[ridx]=observation
        Raction[ridx]=action
        Rreward[ridx]=reward
        Rdone[ridx]=done
        Robs1[ridx]=observation1
        episode.append(ridx)

        rcnt+=1
        if (rcnt> N*5):
            Rfull = True

        if renderFlag:
            env.render()

        episodeReward+=reward
        if done:
            break

        if Rfull:
            sample = np.random.choice(min(rcnt, Rsz), N)
            # update critic
            yq = (Rreward[sample] + gamma * (Qscale * criticp.predict([Robs1[sample], actorp.predict(Robs1[sample])])[:, 0])) / Qscale
            critic.train_on_batch([Robs[sample], Raction[sample]], yq)

            # update the actor
            if i_episode > warmup:
                actions = actor.predict(Robs[sample])
                grads = cgradaf([Robs[sample], actions])[0]
                ya = actions + 0.001 * grads  # nudge action in direction that improves Q
                actor.train_on_batch(Robs[sample], ya)

            # update target networks
            criticp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
            actorp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(actorp.get_weights(), actor.get_weights())])

    RewardsHistory.append(episodeReward)
    print("Episode {} finished after {} timesteps total reward={}".format(i_episode, t + 1, episodeReward))

    if len(episode)>2:
        sp=(6,1)
        plt.clf()
        plt.subplot(*sp,1)
        plt.gca().set_ylim([-1.2,1.2])
        plt.title("Episode {} {}".format(i_episode,"Warming" if (i_episode<warmup) else ""))
        for i in range(Robs[episode].shape[1]):
            plt.plot(Robs[episode, i], label='obs {}'.format(i))
        plt.legend(loc=1)
        plt.subplot(*sp,2)
        plt.gca().set_ylim([1.3*env.action_space.low,1.3*env.action_space.high])
        plt.plot(Raction[episode], 'g', label='action taken')
        actionp=actorp.predict(Robs[episode])
        action=actor.predict(Robs[episode])
        plt.plot(action,'red',label='action')
        plt.plot(actionp,'lightgreen',label='actionp')
        plt.legend(loc=1)
        plt.subplot(*sp,3)
        plt.gca().set_ylim([-15,0])
        plt.plot(Rreward[episode], 'r', label='reward')
        plt.legend(loc=1)
        plt.subplot(*sp,4)
        plt.gca().set_ylim([-15/(1-gamma),50])
        q=Qscale*critic.predict([Robs[episode], Raction[episode]])
        plt.plot(q,'k',label='Q')
        qp=Qscale*criticp.predict([Robs[episode], Raction[episode]])
        plt.plot(qp,'gray',label='Qp')
        discounted_future_reward=Rreward[episode].copy()
        for i in reversed(range(len(discounted_future_reward)-1)):
            discounted_future_reward[i]+= gamma * discounted_future_reward[i + 1]
        plt.plot(discounted_future_reward, 'r', label='R')
        QAccHistory.append(np.mean(np.abs(discounted_future_reward-qp)))
        plt.legend(loc=1)
        plt.subplot(*sp,5)
        plt.gca().set_ylim([-2000,0])
        plt.plot(RewardsHistory, 'r', label='reward history')
        plt.legend(loc=2)
        plt.subplot(*sp,6)
        #plt.gca().set_ylim([-1e3,1e3])
        plt.plot(QAccHistory, 'r', label='Qloss history')
        plt.legend(loc=2)
        plt.pause(0.1)
