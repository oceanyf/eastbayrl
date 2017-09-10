
import numpy as np
import keras.backend as K


def expectation(x):
    print("x {}".format(x.shape))
    s1 = K.sum(x,axis=-2)
    print("s1 {}".format(s1.shape))
    s2 = K.sum(s1,axis=-2)
    print("s2 {}".format(s2.shape))
    s2 = 1.0/s2
    print("s2 {}".format(s2.shape))
    xc = K.variable(np.arange(int(x.shape[1]))/(int(x.shape[1])-1))
    x1 = K.variable(np.ones([x.shape[1]]))
    yc = K.variable(np.arange(int(x.shape[2]))/(int(x.shape[2])-1))
    y1 = K.variable(np.ones([x.shape[2]]))
    xp = K.permute_dimensions(x,(0,3,1,2))

    print("x {}".format(x.shape))
    print("xp {}".format(xp.shape))
    print("xc {}".format(xc.shape))
    print("x1 {}".format(x1.shape))
    print("yc {}".format(yc.shape))
    print("y1 {}".format(y1.shape))

    xx = K.dot(yc, x)
    print("xx {}".format(xx.shape))
    xx = K.dot(x1, xx)
    print("xx {}".format(xx.shape))

    xy = K.dot(y1, x)
    print("xy {}".format(xy.shape))
    xy = K.dot(xc, xy)
    print("xy {}".format(xy.shape))

    nc=K.stack([xx,xy],axis=-1)
    print("nc {}".format(nc.shape))
    nc=K.transpose(K.transpose(nc)*K.transpose(s2))
    print("nc {}".format(nc.shape))
    return nc,s2

#softmax input
a=np.array([
    [
        [[1, 0], [0,0],[0, 0], [0,0]],
        [[0,0],[1,0],[0, 1], [0,1]],
        [[0,0],[0,0],[0,1], [0,1]]],
])
print(a.shape)
#v=K.variable(value=np.array(a))
v=K.placeholder(shape=(None,3,4,2))
nc,s=expectation(v)
func=K.function([v],[nc,s])

print(a)
r=func([a])
print(r)