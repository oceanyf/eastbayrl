from __future__ import division
import numpy as np


# return a dictionary of keyword arguments
#useful for prepackaging arguments
def kwargs(**kwargs):
    return kwargs

# generator for a grid of boxes (x1,y1,x2,y2)
# box arg is output bounding box
def boxgrid(nx,*args,box=(0,0,1,1)):
    ny=nx if not args else args[0]
    for i1,i2 in zip(np.linspace(box[0],box[2],nx+1)[:-1],np.linspace(box[0],box[2],nx+1)[1:]):
        for j1,j2 in zip(np.linspace(box[1],box[3],ny+1)[:-1],np.linspace(box[1],box[3],ny+1)[1:]):
            yield (i1,j1,i2,j2)
def gridsum(a,nx,*args):
    ny=nx if not args else args[0]
    if ((a.shape[-1]%ny)!=0 or (a.shape[-2]%nx)!=0):
        raise ValueError('Array shape must be evenly divisible by grid sizes')
    return np.sum(a.reshape((-1,ny,int(a.shape[-1]//ny))).sum(axis=-1).reshape(-1,nx,int(a.shape[-2]/nx),ny),axis=-2)

# floating point equivalent of range
def frange(x, *args):
    step = 1.0 if len(args)<2 else args[1]
    y = x if len(args)==0 else args[0]
    x = x if len(args)>0 else 0
    while x < y:
        yield x
        x += step

from matplotlib.widgets import CheckButtons


class ToggleFlags:
    def __init__(self,args=None):
        self.names=[]
        self.args=args
    def add(self,name,value=False):
        if name in ['add','showat','__init__']:return
        if name in self.args:
            value=True
        self.names.append(name)
        self.__setattr__(name,value)
    def showat(self,ax):
        v=[self.__getattribute__(name) for name in self.names]
        self.check=CheckButtons(ax,self.names,v)
        def func(label):
            self.__setattr__(label,not self.__getattribute__(label))
            print("clicked")
        self.check.on_clicked(func)
# wrap environment and add UI


