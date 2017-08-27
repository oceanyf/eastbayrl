# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt


Rsz=1000 #replay buffer size
N=128 #mini batch size
tau=0.1
gamma=0.9
Qscale=(1/(1-gamma))
std=0.005

#import project specifics, such as actor/critic models
#can override above hyper parameters
from project import *

print('observation space {} high {} low {}'.format(env.observation_space,env.observation_space.high,env.observation_space.low))
print('action space {} high {} low {}'.format(env.action_space,env.action_space.high,env.action_space.low))
critic.summary()
actor.summary()

#target networks
criticp=keras.models.clone_model(critic)
criticp.compile(optimizer='adam',loss='mse')
criticp.set_weights(critic.get_weights())
actorp=keras.models.clone_model(actor)
actorp.compile(optimizer='adam',loss='mse')
actorp.set_weights(actor.get_weights())

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

plt.ion()
plt.figure(1)

cgradaf = K.function(critic.inputs, K.gradients(critic.outputs, critic.inputs[1]))  # grad of Q wrt actions

for i_episode in range(200000):
    observation1 = env.reset()
    totalReward=0
    clipcnt=0
    run=[]
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
        run.append(ridx)

        ridx= (ridx + 1) % Rsz
        if (ridx == 0):
            Rfull = True

        if(abs(action)>2.0): clipcnt+=1
        #print('done={} action ={}'.format(done,action))
        if (i_episode % 100)==0:
            env.render()
        totalReward+=reward
        if done:
            break

    print("Episode {} finished after {} timesteps total reward={} clipped={}".format(i_episode,t + 1, totalReward,clipcnt))

    if Rfull:
        sample = np.random.choice(Rsz, N)

        # update critic
        # debugging - try fitting the critic without the target
        # yq=Rreward[sample]+gamma*(Qscale*criticp.predict([Robs1[sample],actorp.predict(Robs1[sample])])[:,0])
        yq = (Rreward[sample] + gamma * (Qscale * criticp.predict([Robs1[sample], actorp.predict(Robs1[sample])])[:, 0])) / Qscale
        critic.fit([Robs[sample], Raction[sample]], yq, batch_size=N, epochs=5, verbose=0)

        # update the actor
        actions = actor.predict(Robs[sample])
        grads = cgradaf([Robs[sample], actions])[0]
        ya = actions + 0.01 * grads  # nudge action in direction that improves Q
        ya = np.clip(ya, env.action_space.low, env.action_space.high)
        actor.fit(Robs[sample], ya, verbose=0, epochs=5)

        # update target networks
        criticp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
        actorp.set_weights([tau*w+(1-tau)*wp for wp,w in zip(actorp.get_weights(),actor.get_weights())])

        # diagnostic messages
        q = critic.predict([Robs[sample], actions])
        print('rewards={} grads={} q={}'.format(np.mean(Rreward[sample]), np.mean(grads), np.mean(q)))

    if len(run)>2:
        sp=410
        plt.clf()
        plt.subplot(sp+1)

        plt.title("Episode {}".format(i_episode))
        for i in range(Robs[run].shape[1]):
            plt.plot(Robs[run,i],label='obs {}'.format(i))
        plt.legend(loc=1)
        plt.subplot(sp+2)
        plt.gca().set_ylim([1.3*env.action_space.low,1.3*env.action_space.high])
        plt.plot(Raction[run],'g',label='action taken')
        actionp=actorp.predict(Robs[run])
        action=actor.predict(Robs[run])
        plt.plot(action,'red',label='action')
        plt.plot(actionp,'lightgreen',label='actionp')
        plt.legend(loc=1)
        plt.subplot(sp+3)
        plt.gca().set_ylim([-15,0])
        plt.plot(Rreward[run],'r',label='reward')
        plt.legend(loc=1)
        plt.subplot(sp+4)
        plt.gca().set_ylim([-200,50])
        q=Qscale*critic.predict([Robs[run],Raction[run]])
        plt.plot(q,'k',label='Q')
        qp=Qscale*criticp.predict([Robs[run],Raction[run]])
        plt.plot(qp,'gray',label='Qp')
        r=Rreward[run].copy()
        for i in reversed(range(len(r)-1)):
            r[i]+= gamma*r[i+1]
        plt.plot(r,'r',label='R')
        plt.legend(loc=1)
        plt.pause(0.1)
