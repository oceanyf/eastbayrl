# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt

Rsz=200000 #replay buffer size
N=320 # sample size
tau=0.01
gamma=0.5
warmup=2
renderFlag=False
noiseFlag=True
vizFlag=True
vizIdx=[0, 1]

#import project specifics, such as actor/critic models
from project import *
print("Using {} environment.".format(env.spec.id))
print('observation space {} high {} low {} {} {}'.format(env.observation_space,env.observation_space.high,env.observation_space.low,env.observation_space.shape,env.observation_space.shape[0]))
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


#allocate replay buffers
Robs=np.zeros([Rsz] + list(env.observation_space.shape))
Robs1=np.zeros_like(Robs)
Rreward=np.zeros((Rsz,))
Rdfr=np.zeros((Rsz,))
Raction=np.zeros([Rsz] + list(env.action_space.shape))
Rdone=np.zeros((Rsz,))

#set up the plotting
plt.ion()

def ontype(event): # r  will toggle the rendering, n will togggle noise
    global renderFlag,noiseFlag,vizFlag,env,episodes
    if event.key == 'r' or event.key == ' ':
        renderFlag=not renderFlag
    elif event.key == 'n':
        noiseFlag=not noiseFlag
    elif event.key == 'g':
        env.new_goal()
    elif event.key == 'e':
        episodes=[]
    elif event.key == 'v':
        vizFlag=not vizFlag
plt.gcf().canvas.mpl_connect('key_press_event',ontype)

rcnt=0
Rfull=False
RewardsHistory = []
QAccHistory = []
episodes=[]
for i_episode in range(200000):
    observation1 = env.reset()
    RewardsHistory.append(0)
    episode=[]
    for t in range(1000):
        observation=observation1

        #take step using the action based on actor
        action = actor.predict(np.expand_dims(observation,axis=0))[0]
        if noiseFlag: action += exploration.sample()
        observation1, reward, done, _ = env.step(action)
        if len(observation1.shape)>1 and observation1.shape[-1]==1:
            observation1=np.squeeze(observation1, axis=-1)

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
        if ridx==0: episodes=[] #forget old episodes to avoid wraparound

    if (rcnt > N * 5):
        Rfull = True

    if Rfull:
        for train_iter in range(int(min(rcnt, Rsz)/N)):
            sample = np.random.choice(min(rcnt, Rsz), N)

            # train critic on discounted future rewards
            yq = (Rreward[sample] + gamma * (criticp.predict([Robs1[sample], actorp.predict(Robs1[sample])])[:, 0]))
            critic.train_on_batch([Robs[sample], Raction[sample]], yq)

            # train the actor to maximize Q
            if i_episode > warmup:
                actor.train_on_batch(Robs[sample], np.zeros((N,*actor.output_shape[1:])))

            # update target networks
            criticp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
            actorp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(actorp.get_weights(), actor.get_weights())])
    episodes.append(episode)
    if len(episode)>2:
        fig=plt.figure(1)
        sp=(4,1)
        plt.clf()
        plt.subplot(*sp,1)
        #plt.gca().set_ylim([-1.2,1.2])
        plt.gca().axhline(y=0, color='k')
        fig.suptitle("{}, Episode {} {}{}".format(env.spec.id,i_episode,"Warming" if (i_episode<warmup) else "","/W noise" if noiseFlag else ""))
        for i in range(Robs[episode].shape[1]):
            plt.plot(Robs[episode, i], label='obs {}'.format(i))
        plt.legend(loc=1)
        plt.subplot(*sp,2)
        plt.gca().axhline(y=0, color='k')
        plt.plot(Raction[episode], 'g', label='action taken')
        actionp=actorp.predict(Robs[episode])
        action=actor.predict(Robs[episode])
        plt.plot(action,'red',label='action')
        plt.plot(actionp,'lightgreen',label='actionp')
        plt.legend(loc=1)
        plt.subplot(*sp,3)
        plt.gca().axhline(y=0, color='k')
        plt.plot(Rreward[episode], 'r', label='reward')
        plt.legend(loc=1)
        plt.subplot(*sp,4)
        plt.gca().axhline(y=0, color='k')
        q=critic.predict([Robs[episode], Raction[episode]])
        plt.plot(q,'k',label='Q')
        qp=criticp.predict([Robs[episode], Raction[episode]])
        plt.plot(qp,'gray',label='Qp')
        Rdfr[episode]=Rreward[episode]
        last=0
        for i in reversed(episode):
            Rdfr[i]+= gamma * last
            last=Rdfr[i]

        plt.plot(Rdfr[episode], 'r', label='R')
        QAccHistory.append(np.mean(np.abs(Rdfr[episode]-qp)))
        plt.legend(loc=1)


        #second plot
        plt.figure(2)
        sp=(2,1)
        plt.clf()
        plt.subplot(*sp,1)
        plt.gca().axhline(y=0, color='k')
        plt.plot(RewardsHistory, 'r', label='reward history')
        plt.legend(loc=2)
        plt.subplot(*sp,2)
        plt.gca().axhline(y=0, color='k')
        plt.plot(QAccHistory, 'r', label='Qloss history')
        plt.legend(loc=2)

        #third plot
        if vizFlag:
            fig=plt.figure(3)
            ax = plt.gca()
            plt.clf()
            ax.set_title("Qvalue for obs{}".format(vizIdx))
            #todo: make this a function of the first two action space dimensions
            gsz=100
            oidx0=vizIdx[0]
            oidx1=vizIdx[1]
            ndim=env.observation_space.shape[0]
            nadim=env.action_space.shape[0]
            low=env.observation_space.low
            high=env.observation_space.high
            extent=[low[oidx0],high[oidx0],low[oidx1],high[oidx1]]
            X,Y=np.meshgrid(np.linspace(high[oidx0],low[oidx0],gsz),
                            np.linspace(high[oidx1],low[oidx1],gsz))
            tmp=[]
            for idx in range(ndim):
                if idx in vizIdx:
                    tmp.append(X if idx == vizIdx[0] else Y)
                else:
                    tmp.append(np.ones_like(X) * Robs[0, idx])
            obs = np.array(tmp).T.reshape((gsz*gsz,ndim))
            act=actor.predict(obs)
            A=act.reshape(gsz,gsz,nadim)
            #print("act shape={} A={} act={}, A={}".format(act.shape,A.shape,act[:5],A[1,1]))
            Z = critic.predict([obs,act]).reshape(gsz,gsz)
            vmin=np.min(Z)
            vmax=np.max(Z)
            im = plt.imshow(Z, cmap=plt.cm.RdBu_r, vmin=vmin, vmax=vmax, extent=extent)
            im.set_interpolation('bilinear')
            cb = fig.colorbar(im)
            plt.axis([low[0],high[0],low[1],high[1]])
            for i,e in enumerate(episodes):
                lastone= (i==len(episodes)-1)
                c = 'black' if lastone else 'white'
                s = 6 if lastone else 3
                plt.scatter(x=-Robs[e,1], y=Robs[e,0], cmap=plt.cm.RdBu_r, c=Rdfr[e],vmin=vmin, vmax=vmax,s=s)
                if lastone:
                    plt.scatter(x=-Robs[e,1], y=Robs[e,0], c=c,vmin=vmin, vmax=vmax,s=0.05)
            plt.scatter(x=Robs[episodes[-1][-1],1],y=Robs[episodes[-1][-1],0],c='green',s=s/2)

            fig=plt.figure(4)
            ax = plt.gca()
            plt.clf()
            ax.set_title("Actions for obs{}".format(vizIdx))
            sp=(nadim,1)
            for i in range(nadim):
                plt.subplot(*sp,i+1)
                avmin=env.observation_space.low[i]
                avmax=env.observation_space.high[i]
                im = plt.imshow(A[:,:,i], cmap=plt.cm.RdBu_r, vmin=avmin, vmax=avmax,
                                extent=extent)
                im.set_interpolation('bilinear')
                cb = fig.colorbar(im)

            fig=plt.figure(3)

        plt.pause(0.1)

    print("Episode {} finished after {} timesteps total reward={}".format(i_episode, t + 1, RewardsHistory[-1]))