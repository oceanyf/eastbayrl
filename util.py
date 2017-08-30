import keras.backend as K
def DDPGof(opt):
    class tmp(opt):
        def __init__(self, critic, actor, *args, **kwargs):
            super(tmp, self).__init__(*args, **kwargs)
            self.critic=critic
            self.actor=actor
        def get_gradients(self,loss, params):
            self.combinedloss= -self.critic([self.actor.inputs[0],self.actor.outputs[0]])
            return K.gradients(self.combinedloss,self.actor.trainable_weights)
    return tmp