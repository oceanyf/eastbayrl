# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt

Rsz=200000 #replay buffer size
N=32 # sample size
tau=0.01
gamma=0.95
Oscale=np.array([1,1,.2])
warmup=10


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

#set up the plotting
plt.ion()
plt.figure(1)
renderFlag=True
noiseFlag=True
def ontype(event): # r  will toggle the rendering, n will togggle noise
    global renderFlag,noiseFlag
    if event.key == 'r' or event.key == ' ':
        renderFlag=not renderFlag
    elif event.key == 'n':
        noiseFlag=not noiseFlag
plt.gcf().canvas.mpl_connect('key_press_event',ontype)

rcnt=0
Rfull=False
RewardsHistory = []
QAccHistory = []
lossratio=[]
for i_episode in range(200000):
    observation1 = env.reset()
    RewardsHistory.append(0)
    episode=[]
    for t in range(1000):
        observation=observation1
        #print(observation)
        #take step using the action based on actor
        action = actor.predict(np.expand_dims(observation,axis=0))[0]
        if noiseFlag: action += exploration.sample()
        observation1, reward, done, _ = env.step(action)
        #observation1=observation1[:,0]*Oscale
        #print("obs {} action {} reward {}".format(observation1,action,reward))

        # insert into replay buffer
        ridx=rcnt%Rsz
        rcnt+=1
        Robs[ridx]=observation
        Raction[ridx]=action
        Rreward[ridx]=reward
        Rdone[ridx]=done
        Robs1[ridx]=observation1

        #book keeping
        episode.append(ridx)
        RewardsHistory[-1]+=reward
        if renderFlag:env.render()
        if done: break

    if (rcnt > N * 5):
        Rfull = True

    # test the accuracy of critic.  Disable actor update if critic is not accurate enough

    discounted_future_reward=Rreward[episode].copy()
    for i in reversed(range(len(discounted_future_reward)-1)):
        discounted_future_reward[i]+= gamma * discounted_future_reward[i + 1]
    shortepisode = episode[:int(len(episode) / 2)]
    q=critic.predict([Robs[shortepisode], Raction[shortepisode]])[:,0]
    lossr=np.mean(np.abs(q - discounted_future_reward[:len(q)]))/np.mean(np.abs(discounted_future_reward[:len(q)])+np.abs(q))
    print("qshape {}".format(q.shape))
    print("q {}".format(q[:10]))
    print("R {}".format(discounted_future_reward[:10]))
    print("diff {}".format(q[:10]-discounted_future_reward[:10]))
    print("abs {}".format(np.abs(q[:10]-discounted_future_reward[:10])))
    print("mean abs {}".format(np.mean(np.abs(q[:10]-discounted_future_reward[:10]))))
    print("mean abs R{}".format(np.mean(np.abs(discounted_future_reward[:10]))))
    lossratio.append(lossr)
    trainable=lossratio[-1]<0.1

    print("loss ratio {}".format(lossratio[-1]))
    if Rfull:
        for train_iter in range(len(episode)):
            sample = np.random.choice(min(rcnt, Rsz), N)

            # train critic on discounted future rewards
            yq = (Rreward[sample] + gamma * (criticp.predict([Robs1[sample], actorp.predict(Robs1[sample])])[:, 0]))
            critic.train_on_batch([Robs[sample], Raction[sample]], yq)

            # train the actor to maximize Q
            if i_episode > warmup and trainable:
                actor.train_on_batch(Robs[sample], np.zeros([N,actor.output_shape[1]]))

            # update target networks
            criticp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
            actorp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(actorp.get_weights(), actor.get_weights())])

    if len(episode)>2:
        print("episod length {}".format(len(episode)))
        sp=(7,1)
        plt.clf()
        plt.subplot(*sp,1)
        #plt.gca().set_ylim([-1.2,1.2])
        plt.title("Episode {} {}{} {}".format(i_episode,"Warming" if (i_episode<warmup) else ""
                                           ,"/W noise" if noiseFlag else ""
                                           ,", training actor" if trainable else ", train critic @ {:.2f}".format(lossratio[-1])))
        for i in range(Robs[episode].shape[1]):
            plt.plot(Robs[episode, i], label='obs {}'.format(i))
        plt.legend(loc=1)
        plt.subplot(*sp,2)
        plt.plot(Raction[episode], 'g', label='action taken')
        actionp=actorp.predict(Robs[episode])
        action=actor.predict(Robs[episode])
        plt.plot(action,'red',label='action')
        plt.plot(actionp,'lightgreen',label='actionp')
        plt.legend(loc=1)
        plt.subplot(*sp,3)
        #plt.gca().set_ylim([-15,0])
        plt.plot(Rreward[episode], 'r', label='reward')
        plt.legend(loc=1)
        plt.subplot(*sp,4)
        #plt.gca().set_ylim([-15/(1-gamma),50])
        q = critic.predict([Robs[episode], Raction[episode]])[:,0]
        plt.plot(q, 'k', label='Q')
        qp = criticp.predict([Robs[episode], Raction[episode]])[:,0]
        plt.plot(qp, 'gray', label='Qp')
        QAcc=critic.evaluate([Robs[episode],Raction[episode]],discounted_future_reward)
        QAccHistory.append(QAcc)
        #QAccHistory.append(np.mean(np.abs(discounted_future_reward-qp)))
        plt.plot(discounted_future_reward, 'r', label='R')
        plt.legend(loc=1)
        plt.subplot(*sp,5)
        #plt.gca().set_ylim([-2000,0])
        plt.plot(RewardsHistory, 'r', label='reward history')
        plt.legend(loc=2)
        plt.subplot(*sp,6)
        plt.plot(QAccHistory, 'r', label='Q error history')
        plt.legend(loc=2)
        plt.subplot(*sp,7)
        plt.plot(lossratio, 'r', label='Q loss ratio history')
        plt.legend(loc=2)
        plt.pause(0.1)

    print("Episode {} finished after {} timesteps total reward={}".format(i_episode, t + 1, RewardsHistory[-1]))