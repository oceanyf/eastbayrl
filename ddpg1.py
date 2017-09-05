# from scratch attempt at ddpg with split V+A
#


import gym
import keras
import keras.backend as K
import numpy as np
import argparse
from config import Config
from replay_buffer import ReplayBuffer


def ddpg_training(plt,args=None):

    print("Using {} environment.".format(env.spec.id))
    print('observation space {} '.format(env.observation_space))
    print('action space {} high {} low {}'.format(env.action_space,env.action_space.high,env.action_space.low))
    critic.summary()
    actor.summary()

    # create target networks
    criticp = keras.models.clone_model(critic)
    criticp.compile(optimizer='adam',loss='mse')
    criticp.set_weights(critic.get_weights())
    actorp = keras.models.clone_model(actor)
    actorp.compile(optimizer='adam',loss='mse')
    actorp.set_weights(actor.get_weights())


    #allocate replay buffers
    replay_buffer = ReplayBuffer(Config.buffer_length, env.observation_space.shape, env.action_space.shape)

    #set up the plotting - imports must be here to enable matplotlib.use()
    plt.ion()
    from util import ToggleFlags
    from display import display_progress

    flags=ToggleFlags(args)
    flags.add('noise',True)
    flags.add('render',False)
    flags.add('clear')
    flags.add('viz',True)
    flags.add('movie',True)
    flags.add('trails',False)

    RewardsHistory = []
    Rdfr = np.zeros((Config.buffer_length,))
    episodes = []
    epoches = int(Config.buffer_length / Config.batch_size)

    for i_episode in range(Config.max_episodes):
        observation1 = env.reset()
        episode = []
        for t in range(Config.max_steps):
            episode.append(replay_buffer.index)
            #take step using the action based on actor
            observation = observation1
            action = actor.predict(np.expand_dims(observation, axis=0))[0]
            if flags.noise: action += exploration.sample()
            observation1, reward, done, _ = env.step(action)
            if len(observation1.shape) > 1 and observation1.shape[-1] == 1:
                observation1 = np.squeeze(observation1, axis=-1)

            # insert into replay buffer
            replay_buffer.append(observation, action, reward, observation1, done)

            #book keeping
            RewardsHistory.append(reward)
            if flags.render: env.render()
            if done: break
            if replay_buffer.index == 0: episodes = [] #forget old episodes to avoid wraparound

        if replay_buffer.ready:
            for epoch in range(epoches):
                sample = replay_buffer.sample(Config.batch_size)
                # train critic on discounted future rewards
                yq = (replay_buffer.reward[sample] + Config.gamma * (criticp.predict([replay_buffer.obs1[sample], actorp.predict(replay_buffer.obs1[sample])])[:, 0]))
                critic.train_on_batch([replay_buffer.obs[sample], replay_buffer.action[sample]], yq)

                # train the actor to maximize Q
                if i_episode > Config.warmup:
                    actor.train_on_batch(replay_buffer.obs[sample], np.zeros((Config.batch_size, *actor.output_shape[1:])))

                # update target networks
                criticp.set_weights([Config.tau * w + (1 - Config.tau) * wp for wp, w in zip(criticp.get_weights(), critic.get_weights())])
                actorp.set_weights([Config.tau * w + (1 - Config.tau) * wp for wp, w in zip(actorp.get_weights(), actor.get_weights())])
        if flags.clear:
            episodes = []
        episodes.append(episode)
        if len(episode) > 2 and Config.show_progress:
            display_progress(replay_buffer, flags, plt, RewardsHistory, Rdfr, env, episode, episodes, i_episode, actor, actorp, critic,
                             criticp)
        if Config.save_model and i_episode % 100 == 0:
            print("Save models")
            actor.save('actor.h5')
            critic.save('critic.h5')
        print("Episode {} finished after {} timesteps total reward={}".format(i_episode, t + 1, RewardsHistory[-1]))

if __name__ == "__main__":
    #import project specifics, such as actor/critic models
    from project import *
    parser = argparse.ArgumentParser()
    parser.add_argument('--noscreen',action='store_true')
    args, unknownargs = parser.parse_known_args()
    kwargs={}
    if args.noscreen:
        import matplotlib as matplotlib
        matplotlib.use('Agg')
        kwargs['render']=False
        kwargs['movie']=True
    import matplotlib.pyplot as plt
    ddpg_training(plt,unknownargs)