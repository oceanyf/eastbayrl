import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import cos,sin,sqrt
import random

class NServoArmEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self,**kwargs):
        self.height=160
        self.width=320
        self.max_angle=np.pi
        self.max_speed=1
        self.dt=.05
        self.viewer = None
        self.links=[1,1]

        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(len(self.links),))
        self.image_goal=('image_goal' in kwargs) # NOTE: in this mode a display is required
        if self.image_goal:
            self.height,self.width=kwargs["image_goal"]
            self.channels=3
            print("Image observation {}x{}".format(self.height,self.width))
            self.high = np.array([self.max_angle] * len(self.links) + [255]*self.width*self.height*self.channels)
            self.low = np.array([self.max_angle] * len(self.links) + [0]*self.width*self.height*self.channels)
            self.observation_space = spaces.Box(low=self.low, high=self.high)
        else:
            self.high = np.array([self.max_angle] * len(self.links) + [2, 2])
            self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(len(self.links)+2,))

        self._seed()
        self.state=np.zeros([2+len(self.links)])
        self.linkx=np.zeros_like(self.links)
        self.linky=np.zeros_like(self.links)
        self.linka=np.zeros_like(self.links)
        self.deadband=0.05
        self.lastdr=0
        self.use_random_goals=False
        #self.set_goals([(0,np.sum(self.links))])
        if 'ngoals' in kwargs:
            self.random_goals(int(kwargs['ngoals']))
        else:
            self.random_goals(1)


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
         #move
        for i,l in enumerate(self.links):
            th=self.state[i]
            thdot=np.clip(u[i], -self.max_speed, self.max_speed)
            self.state[i] = th + thdot*self.dt


        # find new position of the end effector
        xs, ys, ts = self.node_pos()

        # determine reward
        d, reward = self.distance_reward(xs,ys)
        reward -= 0.01 # incentive to get something done
        # or np.any(np.greater(u,self.max_speed))
        if np.any(np.less(ys,-0.2)):
            reward -= 2
            self.done=True
        elif d<self.deadband:
            reward += 2
            self.done = True
        return self._get_obs(), reward, self.done, {"goal":(self.goalx,self.goaly),'bounds':self.bounds}

    def _reset(self):
        self.state= np.zeros_like(self.state)
        self.done=False
        if self.use_random_goals: # use a new goal each time
            self.random_goals(1)
        self.goalidx=random.randrange(len(self.goals))
        self.goalx,self.goaly=self.goals[self.goalidx]

        while True: #pick a random but valid state
            self.state = np.random.uniform(-np.pi,np.pi,size=[len(self.links)+2])
            xs,ys,ts = self.node_pos()
            self.distance_reward(xs,ys) # advance the lastdr
            if np.all(np.greater_equal(ys[1:],0.2)):break
        self.state[-2] = self.goalx
        self.state[-1] = self.goaly
        return self._get_obs()

    def _get_obs(self):
        if self.image_goal:
            img = self._render(mode='rgb_array')
            return np.concatenate((angle_normalize(self.state[:len(self.links)]).flatten(), img.flatten()), axis=0)
        else:
            return np.array([angle_normalize(self.state[0]),angle_normalize(self.state[1]),self.state[2],self.state[3]])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width,self.height)
            sz=np.sum(self.links)
            self.bounds=(-1.1*sz,1.1*sz,-0.1*sz,1.1*sz)
            self.viewer.set_bounds(*self.bounds)
            self.pole_transforms=[]

            for i,l in enumerate(self.links):
                rod = rendering.make_capsule(l, 0.2/(i+1))
                rod.set_color(.3, .8, .3)
                transform = rendering.Transform()
                rod.add_attr(transform)
                self.pole_transforms.append(transform)
                self.viewer.add_geom(rod)


            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            goal = rendering.make_circle(0.2)
            goal.set_color(1,0,0)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)

        for px,x,y,t in zip(self.pole_transforms, *self.node_pos()):
            # there is a race condition of some sort, because
            px.set_rotation(t)
            px.set_translation(x,y)

        self.goal_transform.set_translation(self.goalx,self.goaly)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def node_pos(self):
        xs=[0]
        ys=[0]
        ts=[np.pi/2] # zero is straight up
        for i,l in enumerate(self.links):
            ts.append(ts[-1]+self.state[i])
            xs.append(xs[-1]+l*cos(ts[-1]))
            ys.append(ys[-1]+l*sin(ts[-1]))
        del ts[0]
        return xs,ys,ts # xs&ys include end effector

    def random_goals(self,n):
        self.goals=[]
        for i in range(n):
            r = np.sum(self.links)
            r *= np.random.uniform(0.3, 1)
            angle = np.random.uniform(0.2, np.pi-0.2)
            self.goals.append((r * cos(angle),r * sin(angle)))

    def set_goals(self,goals):
        self.goals=goals
        self.reset()
        return

    def distance_reward(self,xs,ys,advance=True):
        d = sqrt((self.goalx-xs[-1])**2+(self.goaly-ys[-1])**2)
        reward = (2*min(1,self.deadband/d) if d!=0.0 else 1 )-self.lastdr
        self.lastdr=d
        return d,reward

    def get_goal_idx(self):
        return self.goalidx


def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

