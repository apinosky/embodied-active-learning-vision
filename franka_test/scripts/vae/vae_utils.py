#!/usr/bin/env python

import numpy as np
import torch
from torch import nn

class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(int(input.size(0)), -1)

class UnFlatten(nn.Module):
    def __init__(self,shape=[5,32,32]):
        super(UnFlatten, self).__init__()
        self.shape=torch.tensor(shape)

    def forward(self, input):
        return input.reshape(int(input.size(0)), int(self.shape[0]), int(self.shape[1]), int(self.shape[2]))

def get_input_dim(img_dim,CNNdict):
    if CNNdict is None: 
        return np.product(img_dim),img_dim
    else:
        new_dim = np.array(img_dim)
        for kernel_size, stride in zip(CNNdict['kernel_size'],CNNdict['stride']):
            new_dim = ( new_dim - (kernel_size-1) - 1 ) // stride +1
        new_dim[0] = CNNdict['channels'][-1]
        # print('get_input_dim, in out',img_dim,new_dim)
        if ('max_pool' in CNNdict.keys()) and (CNNdict['max_pool'] > 0):
            new_dim[1:] = new_dim[1:]//CNNdict['max_pool']
        return np.product(new_dim), new_dim

def get_padding(img_dim,CNNdict):
    if CNNdict is None: 
        return None 
    else: 
        new_dim = np.array(img_dim)
        fwd_dims = [new_dim]
        for kernel_size, stride in zip(CNNdict['kernel_size'],CNNdict['stride']):
            fwd_dims.append((fwd_dims[-1] - (kernel_size-1) - 1 ) // stride +1)

        reverse_dims = [np.zeros(3)]
        for kernel_size, stride, old_dim in zip(reversed(CNNdict['kernel_size']),reversed(CNNdict['stride']),reversed(fwd_dims)):
            reverse_dims.append( (old_dim-1) * stride + (kernel_size-1) +1 )

        padding = []
        for old_dim,new_dim in zip(reversed(fwd_dims),reverse_dims):
            padding.append((old_dim-new_dim)[-1])

        return padding[1:]

def weights_init(seq):
    for m in seq.children():
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_uniform_(m.weight)
            m.reset_parameters()
        elif isinstance(m, nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
            # torch.nn.init.xavier_uniform_(m.weight.data)
            m.reset_parameters()


class reshape1D(nn.Module):
    def __init__(self,img_dim,y_logvar_dim):
        super().__init__()
        self.dim = y_logvar_dim
    def forward(self, x):
        return x.view(-1,self.dim,1,1)

class reshape2D(nn.Module):
    def __init__(self,img_dim,y_logvar_dim):
        super().__init__()
        scale = (np.array(img_dim[1:])/np.array(y_logvar_dim)).astype(int)
        self.up = torch.nn.Upsample(scale_factor = tuple(scale))
        self.y_logvar_dim = y_logvar_dim
    def forward(self, x):
        return self.up(x.view(-1,1,*self.y_logvar_dim))
