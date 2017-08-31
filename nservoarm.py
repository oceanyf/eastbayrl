import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import cos,sin,sqrt

class NServoArmEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_angle=np.pi
        self.max_speed=.2
        self.dt=.05
        self.viewer = None
        self.links=[1,1]
        self.high = np.array([self.max_angle]*len(self.links)+[2,2])
        self.action_space = spaces.Box(low=-self.max_angle, high=self.max_angle, shape=(len(self.links),))
        self.observation_space = spaces.Box(low=-3, high=3, shape=(len(self.links)+2,))
        self._seed()
        self.state=np.zeros([2+len(self.links)])
        self.linkx=np.zeros_like(self.links)
        self.linky=np.zeros_like(self.links)
        self.linka=np.zeros_like(self.links)
        self.goalx=1
        self.goaly=1

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        dt = self.dt

        #u = np.clip(u, -self.max_angle, self.max_angle)

        for i,l in enumerate(self.links):
            th=self.state[i]
            thdot=(u[i]-th)*10 #
            thdot=np.clip(thdot, -self.max_speed, self.max_speed)
            self.state[i] = th + thdot*dt

        x,y=[0,0]
        angle= 0
        for i,l in enumerate(self.links):
            angle+=self.state[i]
            #self.pole_transforms[i].set_rotation(angle)
            #self.pole_transforms[i].set_translation(x,y)
            self.linka[i]=angle
            self.linkx[i]=x
            self.linky[i]=y
            x=x+l*cos(angle)
            y=y+l*sin(angle)

        reward = 2-sqrt((self.goalx-x)**2+(self.goaly-y)**2)

        return self._get_obs(), reward, self.done, {}

    def _reset(self):
        self.state= np.zeros_like(self.state)
        self.done=False
        r=np.sum(self.links)
        #r *= np.random.uniform(0,1)
        angle=np.random.uniform(0,np.pi)
        self.goalx=r*cos(angle)
        self.goaly=r*sin(angle)
        self.state = np.random.uniform(-np.pi,np.pi,size=[len(self.links)+2])
        self.state[-2]=self.goalx
        self.state[-1]=self.goaly
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,250)
            sz=np.sum(self.links)
            self.viewer.set_bounds(-1.1*sz,1.1*sz,-0.1*sz,1.1*sz)
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

            goal = rendering.make_circle(0.02)
            goal.set_color(1,0,0)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)
        angle= 0
        for i,l in enumerate(self.links):
            angle+=self.state[i]
            self.pole_transforms[i].set_rotation(self.linka[i])
            self.pole_transforms[i].set_translation(self.linkx[i],self.linky[i])
        self.goal_transform.set_translation(self.goalx,self.goaly)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)

gym.envs.register(
    id='NServoArm-v0',
    entry_point='nservoarm:NServoArmEnv',
    max_episode_steps=200,
)