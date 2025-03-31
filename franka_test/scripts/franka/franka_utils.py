#!/usr/bin/env python
import numpy as np

def sstows_conversion(pt, ws_lim):
    """ Convert from [0,1] search space limits to franka robot workspace limits"""
    lim = ws_lim[:,1]-ws_lim[:,0]
    new_pt = [pt[i]*lim[i]+ws_lim[i,0] for i in range(len(lim))]
    return new_pt

def wstoss_conversion(pt, ws_lim):
    """ Convert from franka robot workspace limits to  [0,1] search space limits"""
    lim = ws_lim[:,1]-ws_lim[:,0]
    new_pt =[(pt[i]-ws_lim[i,0])/lim[i] for i in range(len(lim))]
    return new_pt

def ws_conversion(pt, in_dim, out_dim):
    """ Convert from [0,1] search space limits to franka robot workspace limits"""
    if len(in_dim.shape) == 1:
        in_dim = in_dim[None,:]
    if len(out_dim.shape) == 1:
        out_dim = out_dim[None,:]
    ilim= in_dim[:,1]-in_dim[:,0]
    olim = out_dim[:,1]-out_dim[:,0]
    N = len(ilim)
    if len(pt.shape) == 1:
        pts = pt[:N]
    else: # allows multiple points to be converted at once
        pts = pt[:,:N]
    new_pt = (pts-in_dim[:,0])/ilim*olim+out_dim[:,0]
    return new_pt

def linearize_yaw(x,states,tray_lim,robot_lim):
    loc = states.rfind('w')
    yaw = ws_conversion(x, tray_lim, robot_lim).T[loc]
    sinW = np.sin(yaw)
    cosW = np.cos(yaw)
    x = x.T
    x[loc] = sinW
    x = np.insert(x,loc+1,cosW,axis=0)
    return x.T

def find_non_vel_locs(states):
    tmp = [[idx,s] for idx,s in enumerate(states) if s==s.lower()]
    non_vel_locs,tmp_states = map(np.stack,zip(*tmp))
    non_vel_states = ''.join(tmp_states.tolist())
    vel_locs = [idx for idx,s in enumerate(states) if s==s.upper()]
    return non_vel_locs, vel_locs, non_vel_states