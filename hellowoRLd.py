# Just test environment runs uring randomly sampled actions
#
import gym
import nservoarm
gym.envs.register(
    id='NServoArm-v0',
    entry_point='nservoarm:NServoArmEnv',
    max_episode_steps=200,
)
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