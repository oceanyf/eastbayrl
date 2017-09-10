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
from movieplot import MoviePlot

print("obs space={}".format(env.observation_space.shape))
everything=make_models(True)
image=everything["image"]
features=everything["features"]
locator=everything["locator"]
clocator=everything["coarse_locator"]
image.compile(optimizer='adam',loss='mse')
features.compile(optimizer='adam',loss='mse')
locator.compile(optimizer=Adam(lr=0.00001), loss='mse')
clocator.compile(optimizer=Adam(lr=0.001), loss='mse')

image.summary()
features.summary()
clocator.summary()
locator.summary()

print("image weights {}".format(np.sum([np.mean(x) for x in image.get_weights()])))
print("cloc weights {}".format(np.sum([np.mean(x) for x in clocator.get_weights()])))

coarse_locator_output_shape=(int(clocator.output_shape[1]), int(clocator.output_shape[2])
                             , int(clocator.output_shape[3]))


#layer = locator.get_layer(name='image_softmax')
#training = K.variable(value=0,name='batch_normalization_1/keras_learning_phase')
#activation=K.function([locator.input,training],[layer.output])

def coarseof(y):
    out = np.zeros((y.shape[0], *coarse_locator_output_shape))
    nx=np.floor((y[:,0]-bounds[0])/(bounds[1]-bounds[0])*coarse_locator_output_shape[0]).astype(int)
    ny=np.floor((y[:,1]-bounds[2])/(bounds[3]-bounds[2])*coarse_locator_output_shape[1]).astype(int)
    for i,(j,k) in enumerate(zip(ny,nx)):
        if i==0:
            print("coarse ijk {} {} {}".format(i,j,k))
        out[i,2-j,k,:]=1
    return out

env.env.use_random_goals=True # 0 means random each time
y=[]
x=[]
eloss=[]
tloss=[]
ctloss=[]
ds=[]
alternate=False
bounds=None
plt.ion()
movie=MoviePlot({1:'pretrain_loss',2:'pretrain_distance',3:'pretrain_activation'})
for i_episode in range(1000000):

    if i_episode % 100 == 0:
        locator.save('locator.h5')
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
        if not bounds: bounds=info['bounds']
        if done: break

        if len(x)>100:
            x = np.array(x)
            y = np.array(y)
            alternate = not alternate
            if alternate:
                coarsey=coarseof(y)
                print("coarsey.shape {}".format(coarsey.shape))
                print("features.shape {}".format(features.output_shape))
                #print("coarsey {}".format(coarsey[:2]))
                ctloss.append(clocator.train_on_batch(np.array(x), coarsey))
                print("coarse training loss={}".format(ctloss[-1]))
                print("image weights {}".format(np.sum([np.mean(x) for x in image.get_weights()])))
                print("cloc weights {}".format(np.sum([np.mean(x) for x in clocator.get_weights()])))
                if True:
                    print("now train locator")
                    loss=locator.train_on_batch(np.array(x), np.array(y))
                    tloss.append(loss)
                    print("training loss={} {}".format(loss,tloss[-1]))

            else:
                loss=locator.evaluate(x,y,verbose=False)
                eloss.append(loss)
                print("eval loss={} {} ".format(eloss[-1],loss))
                yp=locator.predict(x)
                d=np.sqrt(np.square(y[:,0]-yp[:,0])+np.square(y[:,1]-yp[:,1]))
                ds.append(np.mean(d))
                plt.figure(1)
                plt.clf()
                plt.title("Locator Training Loss")
                plt.semilogy(eloss,label='eval')
                plt.semilogy(tloss,label='train')
                plt.semilogy(ctloss,label='Coarse train')
                plt.legend(loc=2)

                plt.figure(2)
                plt.clf()
                plt.title("Mean Error Distance")
                plt.semilogy(ds,label='d')

                plt.figure(3)
                plt.suptitle("softmax activation")
                plt.clf()
                cloc=clocator.predict(np.expand_dims(x[-1],axis=0))
                ir=x[-1,2:].reshape((env.env.height, env.env.width, 3))
                print("ir={}".format(ir.shape))
                nzones=3
                rsum=np.zeros((nzones,nzones))
                hsz=int(env.env.height/nzones)
                wsz=int(env.env.width/nzones)
                cimg=np.zeros_like(ir)
                print("cloc[0,0,0]={}".format(cloc[0,0,0]))
                for h in range(nzones):
                    for w in range(nzones):
                        #print("h {} {} w {} {} ".format(h,h*hsz,w,w*wsz))
                        #print("ir ",h,w,ir[w * wsz:(w + 1) * wsz, h * hsz:(h + 1) * hsz, 1])
                        rsum[h,w]=np.sum(ir[h*hsz:(h+1)*hsz,w*wsz:(w+1)*wsz,1])
                        #cimg[h*hsz:(h+1)*hsz,w*wsz:(w+1)*wsz,:]=np.array([1,1,1])-np.sum(cloc[:,h,w,:],axis=-1)*np.array([0,1,1]) #assume all channel are goal
                        cimg[h * hsz:(h + 1) * hsz, w * wsz:(w + 1) * wsz, :] = cloc[:, h, w, 0]
                print("rsum={}".format(rsum))
                print("bounds={}".format(bounds))
                print("y={}".format(y[-1]))
                print("coarsey={}".format(coarseof(np.expand_dims(y[-1],axis=0))[0,:,:,0]))
                print("Cloc[0]={}".format(cloc[0,:,:,0]))
                activity=features.predict(np.expand_dims(x[-1],axis=0))
                rows= int(activity.shape[-1] / 3 + 2)
                print("rows={} activity.shape={}".format(rows,activity.shape))
                plt.subplot(rows,3, 1)
                plt.imshow(x[-1,2:].reshape(160, 320, 3))
                plt.subplot(rows,3, 3)
                plt.imshow(cimg)
                for i in range(activity.shape[-1]):
                    plt.subplot(rows,3,i+4)
                    plt.imshow(activity[0, :, :, i])

                loc=locator.predict(np.expand_dims(x[-1],axis=0))
                print("loc={}".format(loc))
                plt.subplot(rows,3,2)
                ax=plt.gca()
                ax.margins(ymargin=-0.2)
                ax.set_xlim([bounds[0],bounds[1]])
                ax.set_ylim([bounds[2],bounds[3]])
                ax.scatter(loc[:,0],loc[:,1],c='r',s=6)
                plt.pause(0.1)
                movie.grab_frames()

                print("Episode {:.3f} tloss={:.4f} eloss={:.4f} predict d={:.3f} {} {}".format(i_episode, ctloss[-1], eloss[-1],np.mean(d),y[0],yp[0]))
            locator.save('locator.h5')


            x=[]
            y=[]

def normalize(x,space,shape=None):
    """
    Transform  into 0.0-1.0, or 0-shape
    :param x:
    :param y:
    :param space:
    :param shape:
    :return:
    """

    if shape:
        nx=nx*shape[0]
    return nx
