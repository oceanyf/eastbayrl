# from scratch attempt at ddpg with split V+A
#
import gym
import keras
import keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
from util import ToggleFlags
from config import *
from display import display_progress


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
replay_buffer = {'obs': np.zeros([Rsz] + list(env.observation_space.shape)),
                'obs1': np.zeros([Rsz] + list(env.observation_space.shape)),
                'reward': np.zeros((Rsz,)), 'done': np.zeros((Rsz,)), 'action': np.zeros([Rsz] + list(env.action_space.shape))}

#set up the plotting
plt.ion()

flags=ToggleFlags()
flags.add('noise',True)
flags.add('render',True)
flags.add('clear')
flags.add('viz',True)
flags.add('movie',True)

rcnt=0
Rfull=False
RewardsHistory = []
Rdfr = np.zeros((Rsz,))
episodes=[]

for i_episode in range(200000):
    observation1 = env.reset()
    RewardsHistory.append(0)
    episode = []
    for t in range(1000):
        #take step using the action based on actor
        observation = observation1
        action = actor.predict(np.expand_dims(observation, axis=0))[0]
        if flags.noise: action += exploration.sample()
        observation1, reward, done, _ = env.step(action)
        if len(observation1.shape) > 1 and observation1.shape[-1] == 1:
            observation1 = np.squeeze(observation1, axis=-1)

        # insert into replay buffer
        ridx = rcnt%Rsz
        rcnt += 1
        replay_buffer['obs'][ridx] = observation
        replay_buffer['obs1'][ridx] = observation1
        replay_buffer['action'][ridx] = action
        replay_buffer['reward'][ridx] = reward
        replay_buffer['done'][ridx] = done

        #book keeping
        episode.append(ridx)
        RewardsHistory[-1] += reward
        if flags.render: env.render()
        if done: break
        if ridx==0: episodes=[] #forget old episodes to avoid wraparound

    if (rcnt > N * 5):
        Rfull = True

    if Rfull:
        for train_iter in range(int(min(rcnt, Rsz)/N)):
            sample = np.random.choice(min(rcnt, Rsz), N)

            # train critic on discounted future rewards
            yq = (replay_buffer['reward'][sample] + gamma * (criticp.predict([replay_buffer['obs1'][sample], actorp.predict(replay_buffer['obs1'][sample])])[:, 0]))
            critic.train_on_batch([replay_buffer['obs'][sample], replay_buffer['action'][sample]], yq)

            # train the actor to maximize Q
            if i_episode > warmup:
                actor.train_on_batch(replay_buffer['obs'][sample], np.zeros((N,*actor.output_shape[1:])))

            # update target networks
            criticp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
            actorp.set_weights([tau * w + (1 - tau) * wp for wp, w in zip(actorp.get_weights(), actor.get_weights())])
    if flags.clear:
        episodes=[]
    episodes.append(episode)
    if len(episode) > 2 and showProgress:
        display_progress(replay_buffer, flags, RewardsHistory, Rdfr, env, episode, episodes, i_episode, actor, actorp, critic,
                         criticp)
    if saveModel and i_episode % 100 == 0:
        print("Save models")
        actor.save('actor.h5')
        critic.save('critic.h5')
    print("Episode {} finished after {} timesteps total reward={}".format(i_episode, t + 1, RewardsHistory[-1]))
