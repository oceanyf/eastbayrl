import gym
from gym.spaces import Tuple
import keras
from keras.layers import Dense,Input,BatchNormalization,Dropout,Conv2D,AvgPool2D,Flatten,Lambda
from keras.models import Model
import numpy as np
from keras_util import softmax2d,softargmax,DDPGof,ornstein_exploration


datadir="../data"
modeldir="../data/models/{}"
#tested environments
def make_pendulum():
    return gym.make('Pendulum-v0')
def make_car():
    return gym.make('MountainCarContinuous-v0')
def make_arm():
    import nservoarm
    gym.envs.register(
        id='NServoArm-v0',
        entry_point='nservoarm:NServoArmEnv',
        max_episode_steps=100,
        kwargs={'ngoals': 1,'image_goal':(160,320)}
    )
    env=gym.make('NServoArm-v0')
    return env

env = make_arm()


#create actor,critic
def make_models(everything=False):
    #critic

    ain = Input(shape=env.action_space.shape,name='action')
    oin = Input(shape=env.observation_space.shape, name='observeration')
    if hasattr(env.env,'image_goal'):
        x = Lambda(lambda x: x[:, 2:],name='image_only')(oin)
        iin=keras.layers.Reshape((env.env.height,env.env.width,3))(x)

        # image feature extraction model
        commonkwargs={"activation":'relu'}
        imgin=Input(shape=(env.env.height,env.env.width,3), name='image_model_input')
        x=imgin
        #x=BatchNormalization()(x)
        x=Conv2D(16,(3,3),**commonkwargs)(x)
        x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Dropout(.25)(x)
        #x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Dropout(.25)(x)
        #x=Conv2D(16,(3,3),**commonkwargs)(x)
        x=Conv2D(16,(3,3),**commonkwargs)(x)
        x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Dropout(.25)(x)
        x=BatchNormalization()(x)
        x=Conv2D(16,(3,3),**commonkwargs)(x)
        #x=Conv2D(16,(3,3),activation='softmax',kernel_regularizer="l2",name='image_softmax')
        x=Lambda(softmax2d,name='image_softmax2d')(x)
        image_feature_model=Model(imgin,x,name="Image_feature_extraction_model")

        # feature estimated coordinate model
        x=image_feature_model(iin)
        x = Lambda(softargmax, name="expected_feature_location")(x)
        flat=Flatten(name='flattend_feature')(x)
        x = Dense(2, activation='linear',kernel_regularizer="l2")(flat)
        locatormodel=Model(oin,x)
        #locatormodel.compile(optimizer=Adam(lr=0.001), loss='mse')

        # feature coarse location model
        x=image_feature_model(iin)
        nzones=5
        overlap=0.5
        tmp=image_feature_model.get_output_shape_at(1)
        poolsz=np.floor(np.array((tmp[-3],tmp[-2]))/(nzones)) #size of grid
        stridesz=poolsz*(1-overlap)
        coarse_map=AvgPool2D(pool_size=poolsz,padding='same',name="coarse_map")(x)
        coarse_locator=Model(oin,coarse_map)
        #locatormodel.compile(optimizer=Adam(lr=0.001), loss='mse')

        # feature activation model
        x=image_feature_model(iin)
        features=Model(oin,x) # just used to display activation levels. Never trained
        #features.compile(optimizer=Adam(lr=0.001), loss='mse')

        sensors = Lambda(lambda x: x[:,:2],name='sensors_only')(oin)
        cin = keras.layers.concatenate([sensors,flat],name='sensor_image')
    else:
        cin = keras.layers.concatenate([oin, ain],name='sensor')

    x=keras.layers.concatenate([cin, ain], name='sensor_action')
    x=Dense(64, activation='relu')(x)
    #x=Dropout(.5)(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu')(x)
    x=Dense(64, activation='relu',kernel_regularizer='l2')(x)
    #x=Dropout(.5)(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu')(x)
    x=Dense(32, activation='relu',kernel_regularizer='l2')(x)
    x=Dense(1, activation='linear', name='Q')(x)
    critic=Model([oin,ain], x)
    #critic.compile(optimizer=Adam(lr=0.1),loss='mse')

    #actor
    x=cin
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(32,activation='relu',kernel_regularizer='l2')(x)
    x=Dense(16,activation='relu',kernel_regularizer='l2')(x)
    #x=Dropout(.5)(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(16,activation='relu')(x)
    x=Dense(env.action_space.shape[0],activation='linear')(x)
    actor=Model(oin, x)
    #actor.compile(optimizer=DDPGof(Adam)(critic, actor, lr=0.001), loss='mse')
    if everything:
        return {'actor':actor, 'critic':critic,'locator':locatormodel,
                'coarse_locator':coarse_locator,'features':features,
                'image':image_feature_model}
    else:
        return actor,critic

actor,critic=make_models()

#exploration policy
exploration=ornstein_exploration(env.action_space,theta=.5, mu=0., dt=1.0,)
