#pretrain the image model
import nservoarm
import gym
import numpy as np
from math import cos,sin,sqrt
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Conv2D,MaxPool2D,Flatten,Lambda
from keras.models import Model
from keras.optimizers import Adam


from project import *

print("obs space={}".format(env.observation_space.shape))
actor,critic,locator = make_models(locator=True)
locator.compile(optimizer=Adam(lr=0.001), loss='mse')

locator.summary()
env.env.use_random_goals=True # 0 means random each time
y=[]
x=[]
eloss=[]
tloss=[]
alternate=False
plt.ion()
for i_episode in range(10000):
    #print("Episode {} ".format(i_episode))
    observation1 = env.reset()
    for t in range(10):
        # take step using the action based on actor
        observation = observation1
        #action = actor.predict(np.expand_dims(observation, axis=0))[0]
        action = env.env.action_space.sample()
        #if flags.noise: action += exploration.sample()
        observation1, reward, done, info = env.step(action)
        observation1 /= 255.0
        y.append(info["goal"])
        x.append(observation1)
        if done: break

        if len(x)>100:
            x = np.array(x)
            y = np.array(y)
            alternate = not alternate
            if alternate:
                tloss.append(locator.train_on_batch(np.array(x),np.array(y)))
            else:
                eloss.append(locator.evaluate(x,y,verbose=False))

                plt.figure(1)
                plt.clf()
                plt.title("Locator Training Loss")
                plt.semilogy(eloss,label='eval')
                plt.semilogy(tloss,label='train')
                plt.legend(loc=2)

                plt.figure(2)
                img = observation1[2:].reshape(160, 320, 3)
                plt.imshow(img)
                plt.pause(0.1)
                xp=x[0:1]
                yp=locator.predict(xp)
                d=sqrt((y[0,0]-yp[0,0])**2+(y[0,1]-yp[0,1])**2)
                print("Episode {:.3f} tloss={:.4f} eloss={:.4f} predict d={:.3f} {} {}".format(i_episode, tloss[-1], eloss[-1],d,y[0],yp[0]))

            x=[]
            y=[]