# Just test environment runs uring randomly sampled actions
#
import gym
import nservoarm

env = gym.make('NServoArm-v0')
for i_episode in range(200):
    observation = env.reset()
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(reward,observation)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break