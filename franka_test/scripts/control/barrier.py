#!/usr/bin/env python
import torch
import numpy as np
import yaml
from argparse import Namespace
import sys, os
base_path = os.path.dirname(os.path.abspath(__file__))


def setup_barrier(states,robot_lim,robot_ctrl_lim,non_vel_locs,dtype,extra_args,rot_states=False): 
    ## load config
    with open(base_path+'/robot_config.yaml') as f:
        yaml_config = yaml.load(f,Loader=yaml.FullLoader)
    self = Namespace()
    for k, v in yaml_config.items():
        setattr(self, k, v)
    ## process
    robot_lim = np.array(robot_lim.copy())[non_vel_locs]
    barr_lim = robot_lim.tolist() + robot_ctrl_lim.tolist() # for barrier function
    if self.use_barrier:
        power = [2. if key in 'rp' else 4. for key in states.lower()]*2
        if self.position_barrier and not self.velocity_barrier: # states only
            barr_weight = [self.barr_weight]*len(states) +  [0]*len(states) 
        elif self.velocity_barrier and not self.position_barrier : # velocity only
            barr_weight = [0]*len(states) + [self.barr_weight]*len(states)
        else: 
            barr_weight = self.barr_weight
        barrier = BarrierFunction(b_lim = barr_lim, barr_weight=barr_weight, b_buff=0.1, power=power)
        # cprint(self.barrier,'cyan')
        # if rot_states:
        #     barrier = TiltBarrierFunction(self.barrier,self.states,tilt_lim=3.1,**extra_args)
    else:
        barrier = NoBarrier()
    return barrier,barr_lim


class BarrierFunction(object):

    def __init__(self, b_lim, power=4, barr_weight=100.0, b_buff=0.01):
        ''' Barrier Function '''
        self.ergodic_dim = len(b_lim)
        self.b_buff = b_buff

        if not(isinstance(power,list)):
            power = [power]*self.ergodic_dim
        self.power = np.array([power]).T

        if not(isinstance(barr_weight,list)):
            barr_weight = [barr_weight]*self.ergodic_dim
        self.barr_weight = np.array([barr_weight]).T

        self.update_lims(b_lim)
    
    def update_lims(self,b_lim):
        self.b_lim = np.copy(b_lim)
        for i in range(self.ergodic_dim):
            self.b_lim[i][0] = b_lim[i][0] + self.b_buff
            self.b_lim[i][1] = b_lim[i][1] - self.b_buff

    def update_ergodic_dim(self,new_ergodic_dim): 
        self.ergodic_dim = new_ergodic_dim
        self.b_lim = self.b_lim[:new_ergodic_dim]
        self.barr_weight = self.barr_weight[:new_ergodic_dim]
        self.power = self.power[:new_ergodic_dim]

    def barr(self,x):
        x_check = x[:self.ergodic_dim]
        boundary = np.stack([(x_check <= self.b_lim[:,0]), # lower
                                (x_check >= self.b_lim[:,1])]).T.astype(int)# upper
        barr_temp = self.barr_weight*(x_check[:,None]-self.b_lim)**self.power
        return np.sum(boundary*barr_temp)

    def dbarr(self, x):
        dbarr_out = np.zeros_like(x)
        x_check = x[:self.ergodic_dim]
        boundary = np.stack([(x_check <= self.b_lim[:,0]), # lower
                                (x_check >= self.b_lim[:,1])]).T.astype(int)# upper
        dbarr_temp = self.power*self.barr_weight*(x_check[:,None]-self.b_lim)**(self.power-1)
        dbarr_out[:self.ergodic_dim] = np.sum(boundary*dbarr_temp,1)
        return dbarr_out

    def __call__(self,x):
        return np.array([self.barr(xt) for xt in x])

    def __repr__(self):
        return  'BarrierFunction('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'

def dummy_conversion(x):
    return x


class TiltBarrierFunction(object):

    def __init__(self, other_bar, states, tilt_lim, tilt_power=4,tilt_weight=10.0,pitch_control=False,rot_to_angles_fn=None,angles_to_rot_fn=None):
        ''' Tilt Barrier Function '''
        self.r_idx = states.rfind('r')
        self.p_idx  = states.rfind('p')
        self.b_lim = tilt_lim
        self.power = tilt_power
        self.weight = tilt_weight
        self.other_bar = other_bar

        self.w_idx = states.rfind('w')
        self.w_b_lim = other_bar.b_lim[self.w_idx].copy()

        self.rot_to_angles_fn = rot_to_angles_fn if rot_to_angles_fn is not None else dummy_conversion
        self.rpw = np.array([self.r_idx,self.p_idx,self.w_idx],dtype=int)

    def update_lims(self,b_lims): 
        self.other_bar.update_lims(b_lims)

    def update_ergodic_dim(self,new_ergodic_dim): 
        self.other_bar.update_ergodic_dim(new_ergodic_dim)

    def barr(self,x):
        barr_temp = 0.
        rot = self.rot_to_angles_fn(x[self.rpw])
        tilt = np.arccos(np.cos(rot[0])*np.cos(rot[1]))
        self.other_bar.b_lim[self.w_idx] = tilt/np.pi*self.w_b_lim.copy()
        if tilt <= self.b_lim:
            barr_temp += self.weight*(tilt-self.b_lim)**self.power
        return barr_temp + self.other_bar.barr(x)

    def dbarr(self, x):
        dbarr_temp = np.zeros_like(x)
        rot = self.rot_to_angles_fn(x[self.rpw])
        r,p,w = rot
        tilt = np.arccos(np.cos(r)*np.cos(p))
        self.other_bar.b_lim[self.w_idx] = tilt/np.pi*self.w_b_lim.copy()
        # print(tilt)
        if tilt <= self.b_lim:
            # print('tilt_low',tilt)
            dbarr_temp[self.r_idx] += self.power*self.weight*(tilt-self.b_lim)**(self.power - 1
                                        )*np.sin(r)*np.cos(p)/np.sqrt(-np.cos(p)**2*np.cos(r)**2 + 1)
            dbarr_temp[self.p_idx] += self.power*self.weight*(tilt-self.b_lim)**(self.power - 1
                                        )*np.sin(p)*np.cos(r)/np.sqrt(-np.cos(p)**2*np.cos(r)**2 + 1)
        return dbarr_temp + self.other_bar.dbarr(x)

    def __call__(self,x):
        return np.array([self.barr(xt) for xt in x])

    def __repr__(self):
        return  'BarrierFunction('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'


class NoBarrier(object):
    def __init__(self):
        pass
    def barr(self,x,others=None):
        return 0.
    def dbarr(self,x,others=None):
        return np.zeros(len(x))
    def __call__(self,x,others=None):
        return np.zeros(len(x))
    def __repr__(self):
        return  'NoBarrier (Dummy Function)'


class VelocityBarrier(object):

    def __init__(self, planner_states, b_lim=0.1, power=4, barr_weight=100.0):
        ''' Barrier Function '''
        self.planner_states = planner_states
        self.skip = [True if s.lower() == s else False for s in planner_states] # only use velocities
        self.state_dim = len(planner_states)

        if not(isinstance(power,list)):
            power = [power]*self.state_dim
        self.power = power

        if not(isinstance(barr_weight,list)):
            barr_weight = [barr_weight]*self.state_dim
        self.barr_weight = barr_weight

        if isinstance(b_lim,float):
            b_lim = np.tile(np.array([[-1.,1.]]),(self.state_dim,1))*b_lim
        self.b_lim = b_lim

    def barr(self,x_new,x_old):
        barr_temp = 0.
        b_lim_tmp = x_old[:,None] + self.b_lim
        for i, (weight,power,b_lim,skip) in enumerate(zip(self.barr_weight,self.power,b_lim_tmp,self.skip)):
            if skip:
                pass
            elif x_new[i] >= b_lim[1]:
                barr_temp += weight*(x_new[i] - (b_lim[1]))**power
            elif x_new[i] <=  b_lim[0]:
                barr_temp += weight*(x_new[i] - (b_lim[0]))**power
        return barr_temp

    def dbarr(self, x_new,x_old):
        dbarr_temp = np.zeros(len(x_new))
        b_lim_tmp = x_old[:,None] + self.b_lim
        for i, (weight,power,b_lim,skip) in enumerate(zip(self.barr_weight,self.power,b_lim_tmp,self.skip)):
            if skip:
                pass
            elif x_new[i] >= b_lim[1]:
                dbarr_temp[i] += power*weight*(x_new[i] - (b_lim[1]))**(power-1)
            elif x_new[i] <=  b_lim[0]:
                dbarr_temp[i] += power*weight*(x_new[i] - (b_lim[0]))**(power-1)
        return dbarr_temp

    def __call__(self,x_old,x_new):
        return np.array([self.barr(xt_old,xt_new) for xt_old,xt_new in zip(x_old,x_new)])

    def __repr__(self):
        return  'VelocityBarrier('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'
