#!usr/bin/env python

### Python Imports ###
import torch
import numpy as np

def psi_fn(traj, samples, std, nu):
    inner = torch.square(traj-samples)/std
    psi = torch.exp(-0.5 * torch.sum(inner, 2))
    return psi / nu

def dpsi_dx_fn(x_explr, samples, std, nu):
    diff = -(x_explr-samples)/torch.abs(std)
    psi = psi_fn(x_explr.unsqueeze(0).unsqueeze(0),samples.unsqueeze(1), torch.abs(std),nu)
    return diff*psi 

def traj_footprint_vec(traj, samples, explr_idx, std, nu):
    """ time-averaged footprint of trajectory (used to calc q)"""
    traj_explr = traj[:,explr_idx] # assumes traj is a set of states (x) 
    psi = psi_fn(traj_explr.unsqueeze(0),samples.unsqueeze(1), torch.abs(std),nu)
    pdf = torch.sum(psi, 1)
    return pdf 

def traj_spread_vec(traj, samples, explr_idx, std, nu):
    """ time-averaged footprint of trajectory (used to calc q)"""
    traj_explr = traj[:,explr_idx] # assumes traj is a set of states (x) 
    psi = psi_fn(traj_explr.unsqueeze(0),samples.unsqueeze(1), torch.abs(std),nu)
    pdf = torch.amax(psi, 1)
    return pdf 

def kldiv_grad_vec(x,samples, explr_idx, std, importance_ratio, nu):
    """ gradient of state footprint; grad is d_psi/d_x; psi is the part inside the integral of q"""
    grad = torch.zeros((samples.shape[0],x.shape[0]),dtype=x.dtype)
    grad[:,explr_idx] = dpsi_dx_fn(x[explr_idx], samples, std, nu) # assumes x is a single state
    grad = torch.sum(grad*importance_ratio.unsqueeze(1),dim=0)
    return grad

def cost_norm(dist): 
    dist[torch.isnan(dist)] = 1e-6
    dist /= torch.sum(dist)
    # dist = torch.clamp(dist,1e-6,None)
    return dist


def renormalize(dist,dim=None,min_val=1e-6):
    if dim is not None:
        dist = dist / torch.sum(dist,dim,keepdims=True)
        dist = torch.clamp(dist,min_val,None)
        dist = torch.log(dist)
        dist = dist - torch.max(dist,dim,keepdims=True)
        dist = torch.exp(dist)
    else:
        dist = dist / torch.sum(dist)
        dist = torch.clamp(dist,min_val,None)
        dist = torch.log(dist)
        dist = dist - torch.max(dist)
        dist = torch.exp(dist)
    return dist

class Lambda(torch.nn.Module):
    "An easy way to create a pytorch layer for a simple `func`."
    def __init__(self, func, vars):
        "create a layer that simply calls `func` with `x`"
        super(Lambda,self).__init__()
        self.func=func
        self.vars=vars
    def forward(self, x): return self.func(x,*self.vars)
    def extra_repr(self): 
        return self.func.__name__ + '(x,*vars)' + '\n vars = ' + '\n\t'.join([str(v) for v in self.vars])
