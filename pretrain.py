#pretrain the image model
import nservoarm
import gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Conv2D,MaxPool2D,Flatten,Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as K


from project import *

print("obs space={}".format(env.observation_space.shape))
actor,critic,locator = make_models(locator=True)
locator.compile(optimizer=Adam(lr=0.001), loss='mse')

locator.summary()
layer = locator.get_layer(name='image_softmax')
training = K.variable(value=0,name='batch_normalization_1/keras_learning_phase')

activation=K.function([locator.input,training],[layer.output])

env.env.use_random_goals=True # 0 means random each time
y=[]
x=[]
eloss=[]
tloss=[]
ds=[]
alternate=False
plt.ion()
for i_episode in range(1000000):
    #print("Episode {} ".format(i_episode))
    observation1 = env.reset()
    for t in range(5):
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
                yt=y
            else:
                eloss.append(locator.evaluate(x,y,verbose=False))

                yp=locator.predict(x)
                d=np.sqrt(np.square(y[:,0]-yp[:,0])+np.square(y[:,1]-yp[:,1]))
                ds.append(np.mean(d))
                plt.figure(1)
                plt.clf()
                plt.title("Locator Training Loss")
                plt.semilogy(eloss,label='eval')
                plt.semilogy(tloss,label='train')
                plt.legend(loc=2)

                plt.figure(2)
                plt.clf()
                plt.title("Mean Error Distance")
                plt.semilogy(ds)

                plt.figure(3)
                plt.suptitle("softmax activation")
                lp=K.learning_phase()
                print("Learning phase {}".format(lp))
                K.set_learning_phase(0)
                weights=activation([x[-1:]])[0]
                K.set_learning_phase(1)
                rows=int(weights.shape[-1])/3+1
                plt.subplot(rows,3, 1)
                plt.imshow(observation1[2:].reshape(160, 320, 3))
                for i in range(weights.shape[-1]):
                    plt.subplot(rows,3,i+2)
                    plt.imshow(weights[0,:,:,i])

                plt.pause(0.1)

                print("Episode {:.3f} tloss={:.4f} eloss={:.4f} predict d={:.3f} {} {}".format(i_episode, tloss[-1], eloss[-1],np.mean(d),y[0],yp[0]))
            locator.save('locator.h5')
            if i_episode % 100 == 0:
                locator.save('locator.h5')
                #with open('locator.pickle', 'wb') as f:
                #    pickle.dump(locator, f)
            x=[]
            y=[]