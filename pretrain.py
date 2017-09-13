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
from util import gridsum,frange

llr=0.001
cllr=0.000

from project import *
from movieplot import MoviePlot

print("obs space={}".format(env.observation_space.shape))
everything=make_models(True)
image=everything["image"]
features=everything["features"]
locator=everything["locator"]
sam=everything['softargmax']
clocator=everything["coarse_locator"]
image.compile(optimizer='adam',loss='mse')
features.compile(optimizer='adam',loss='mse')
locator.compile(optimizer=Adam(lr=llr), loss='mse')
clocator.compile(optimizer=Adam(lr=cllr), loss='mse')
sam.compile(optimizer=Adam(lr=llr), loss='mse')

image.summary()
features.summary()
clocator.summary()
locator.summary()

print("load weights")
#locator.load_weights("./movies/CCCCCCCBelu/locator.h5")
locator.load_weights("locator.h5")

cloc_shape=(int(clocator.output_shape[1]), int(clocator.output_shape[2]), int(clocator.output_shape[3]))
nfeatures=int(features.output_shape[3])
image_shape=(int(image.input_shape[1]), int(image.input_shape[2]), 3)
nzones=int(clocator.output_shape[1])
print("images shape={}".format(image_shape))

poolingfactor= features.output_shape[2] * features.output_shape[1] / (nzones * nzones)

# This generates a target coarse locator output by counting red pixels in the image
def coarseygroundtruth(x):
    ncfeatures=clocator.output_shape[-1]
    ximg=x[:, 2:].reshape((-1, *image_shape))
    coarsey = maxpoolcolor(ximg, [1, 0, 0], nzones)
    coarsey = np.divide(coarsey, coarsey.sum(axis=(-1, -2))[:, None, None])  # normalize
    coarsey = np.repeat(coarsey, ncfeatures, axis=-1)  # copy for each locator channel
    coarsey = coarsey.reshape((-1, nzones, nzones, ncfeatures))
    return coarsey

#assume img[batch,x,y,c]
def maxpoolcolor(img,color,nzonesx,*args):
    nzonesy=nzonesx if not args else args[0]
    cimg=np.equal(img,color).all(axis=-1)
    return gridsum(cimg,nzonesx,nzonesy)

def img_add_grid(img,nzonesx,*args,rgb=None,width=1,value=1):
    nzonesy=nzonesx if not args else args[0]
    if not rgb: rgb = (img.shape[-1]==3)
    ix,iy= (-3,-2) if rgb else (-2,-1)
    if rgb:
        for x in frange(0,img.shape[ix],img.shape[ix]/nzonesx):
            img[...,int(x):int(x)+width,:,:] = value
        for y in frange(0, img.shape[iy], img.shape[iy]/nzonesy):
            img[...,:,int(y):int(y)+width,:] = value
    else:
        for x in frange(0,img.shape[ix],img.shape[ix]/nzonesx):
            img[...,int(x):int(x)+width,:] = value
        for y in frange(0, img.shape[iy], img.shape[iy]/nzonesy):
            img[...,:,int(y):int(y)+width] = value
env.env.use_random_goals=True # 0 means random each time
y=[]
x=[]
eloss=[]
tloss=[]
ctloss=[]
ds=[]
alternate=False
bounds=None
def i2d(x):
    return x*np.array([(bounds[1] - bounds[0]),(bounds[3] - bounds[2])])+  np.array([bounds[0],bounds[2]])
def d2i(x):
    return (x - np.array([bounds[0],bounds[2]]))/np.array([(bounds[1] - bounds[0]),(bounds[3] - bounds[2])])

plt.ion()
fig=plt.figure(3)
fig.set_size_inches(8, 4.5)

movie=MoviePlot({1:'pretrain_loss',2:'pretrain_distance',3:'pretrain_activation'})
for i_episode in range(1000000):
    locked=False # i_episode<10000

    if i_episode % 100 == 0:
        print("Saving locator")
        locator.save('locator.h5')
        locator.save_weights('locator_weights.h5')

    observation1 = env.reset()
    for t in range(5):
        # take step using the action based on actor
        observation = observation1
        #action = actor.predict(np.expand_dims(observation, axis=0))[0]
        action = env.env.action_space.sample()
        #if flags.noise: action += exploration.sample()
        observation1, reward, done, info = env.step(action)
        observation1 /= 255.0
        if not bounds: bounds=info['bounds']
        #y.append(d2i(list(info["goal"])))
        y.append(info["goal"])
        x.append(observation1)

        if done: break
        if len(x)>100:
            x = np.array(x)
            y = np.array(y)
            alternate = not alternate
            if alternate:
                if cllr != 0.0:
                    coarsey=coarseygroundtruth(x)
                    ctloss.append(clocator.train_on_batch(np.array(x), coarsey/poolingfactor))
                else:
                    ctloss.append(None)
                if not locked:
                    loss=locator.train_on_batch(np.array(x), d2i(np.array(y)))
                    tloss.append(loss)
                else:
                    tloss.append(None)
                print("coarse training loss={} image weights {} training loss={} ".format(ctloss[-1],
                    np.sum([np.mean(x) for x in clocator.get_weights()]),"locked" if locked else tloss[-1] ))
            else:
                loss=locator.evaluate(x,d2i(y),verbose=False)
                eloss.append(loss)
                yp=i2d(locator.predict(x))
                print(yp.shape)
                d=np.sqrt(np.square(y[:,0]-yp[:,0])+np.square(y[:,1]-yp[:,1]))
                ds.append(np.mean(d))


                # display status and trends
                fig=plt.figure(1)
                plt.clf()
                plt.title("Locator Training Loss - Episode {} {}".format(i_episode,"locked" if locked else ""))
                #fig, ax1 = plt.subplots()
                ax1=plt.gca()
                ax1.semilogy(eloss,label='eval')
                ax1.semilogy(tloss,label='train')
                plt.legend(loc=2)
                ax2 = ax1.twinx()
                ax2.semilogy(ctloss,label='Coarse train',c='green')
                fig.tight_layout()
                plt.legend(loc=1)

                plt.figure(2)
                plt.clf()
                plt.title("Mean Error Distance - Episode {}".format(i_episode))
                plt.semilogy(ds,label='d')

                plt.figure(3)
                fig = plt.gcf()
                #fig.subplots_adjust(hspace=1.5)
                plt.suptitle("softmax activation")
                plt.clf()
                cols=4
                rows= int(features.output_shape[-1] / cols + 1)
                plt.suptitle("Activation level - Episode {} LR={}/{}".format(i_episode,llr,cllr))

                # run models on last observation
                cloc=clocator.predict(np.expand_dims(x[-1],axis=0))
                cloc=np.mean(cloc,axis=-1,keepdims=True) # average over all feature channels
                cloc*=poolingfactor # scale up by pooling factor
                groundtruth=coarseygroundtruth(x[-1:])
                loc=i2d(locator.predict(np.expand_dims(x[-1],axis=0)))
                samactivity=i2d(sam.predict(np.expand_dims(x[-1],axis=0)))
                print("act loc={} est loc={}".format(y[-1],loc))
                print("act cloc={}".format(np.mean(groundtruth,axis=-1)))
                print("est cloc[0]={}".format(np.mean(cloc[0,:,:],axis=-1)))

                # copy of image observation
                plt.subplot(rows,cols, 1)
                plt.gca().set_title('Observation'.format())
                plt.axis('off')
                plt.grid(True)
                plt.gca().set_aspect('equal')
                plt.imshow(x[-1,2:].reshape(image_shape),extent=bounds) # image observation

                # plot estimated location
                plt.subplot(rows,cols,2)
                ax=plt.gca()
                #plt.axis('off')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                #plt.grid(True)
                ax.margins(ymargin=-0.2)
                ax.set_xlim([bounds[0],bounds[1]])
                ax.set_ylim([bounds[2],bounds[3]])
                ax.set_aspect('equal')

                #print("goal {}".format(y[-1]))
                for i in range(samactivity.shape[1]):
                    #print("sam {}".format(samactivity[0,i]))
                    ax.scatter(*samactivity[0, i], c='b', s=2)
                ax.scatter(loc[0, 0], loc[0, 1], c='g', s=6)
                ax.scatter(y[-1, 0], y[-1, 1], c='r', s=3)
                ax.set_title('Est Location'.format())

                # coarse estimated location
                plt.subplot(rows, cols, 3)
                #cimg=np.repeat(np.sum(cloc, axis=-1), 3, axis=-1).reshape((-1, nzones, nzones, 3))
                cimg=np.mean(cloc,axis=-1)
                plt.gca().set_title('Coarse Est'.format())
                plt.axis('off')
                plt.imshow(cimg[0],aspect=image_shape[0]/image_shape[1],vmin=0,vmax=1.0)

                # coarse location truth
                plt.subplot(rows, cols, 4)
                #gimg = np.repeat(np.sum(groundtruth, axis=-1), 3, axis=-1).reshape((-1,nzones,nzones,3))/nfeatures
                gimg = np.mean(groundtruth, axis=-1)
                plt.gca().set_title('Coarse Truth'.format())
                plt.axis('off')
                plt.imshow(gimg[0],aspect=image_shape[0]/image_shape[1],vmin=0,vmax=1.0)

                # softmax activation levels
                activity=features.predict(np.expand_dims(x[-1],axis=0))
                vmax=np.max(activity[0,:,:,:])
                vmin=np.min(activity[0,:,:,:])
                for i in range(activity.shape[-1]):
                    #plt.gca().set_title('Feature {}'.format(i))
                    plt.subplot(rows,cols,i+5)
                    ax2=plt.gca()
                    plt.axis('off')
                    img=activity[0, :, :, i]
                    #img_add_grid(img, nzones,rgb=False)
                    plt.imshow(img,vmin=vmin,vmax=vmax,extent=bounds)
                    ax2.set_xlim([bounds[0], bounds[1]])
                    ax2.set_ylim([bounds[2], bounds[3]])
                    ax2.set_aspect('equal')
                    ax2.scatter(*samactivity[0, i], c='w', s=6 ,alpha=0.2)
                    ax2.scatter(*y[-1], c='r', s=3, alpha=0.2)



                plt.pause(0.1)
                movie.grab_frames()

                print("Episode {:.3f} predict d={:.3f} act {} est {}".format(i_episode,np.mean(d),y[-1],yp[-1]))
            locator.save('locator.h5')


            x=[]
            y=[]
