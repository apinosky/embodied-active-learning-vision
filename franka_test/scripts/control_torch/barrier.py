#!/usr/bin/env python
import torch
import yaml
from argparse import Namespace
import sys, os
base_path = os.path.dirname(os.path.abspath(__file__))

def setup_barrier(states,robot_lim,robot_ctrl_lim,non_vel_locs,dtype,extra_args,rot_states=False,override_use_barrier=False,uniform=False): 
    if uniform: 
        file =  '/robot_config_uniform.yaml'
    else: 
        file = '/robot_config.yaml'
    ## load config
    with open(base_path+file) as f:
        yaml_config = yaml.load(f,Loader=yaml.FullLoader)
    self = Namespace()
    if override_use_barrier: 
        self.use_barrier = True
    for k, v in yaml_config.items():
        setattr(self, k, v)
    ## process
    barr_lim = torch.tensor(robot_lim[non_vel_locs].tolist() + robot_ctrl_lim.tolist(),dtype=dtype) # for barrier function
    if self.use_barrier:
        # power = [2. if key in 'rp' else 4. for key in states.lower()]*2
        power = [4. for key in states]*2
        if self.position_barrier and not self.velocity_barrier: # states only
            barr_weight = [self.barr_weight]*len(states) + [0]*len(states) 
        elif self.velocity_barrier and not self.position_barrier: # velocity only
            barr_weight = [0]*len(states) + [self.barr_weight]*len(states)
        else: 
            barr_weight = self.barr_weight    
        barrier = BarrierFunction(b_lim = barr_lim, barr_weight=barr_weight, b_buff=0.1, power=power)
        # if rot_states:
            # self.barrier = TiltBarrierFunction(self.barrier,self.states,tilt_lim=2.45,**extra_args) # 3.1; deg_tilt = 180-tilt_lim/pi*180 
    else:
        barrier = NoBarrier()
    return barrier, barr_lim


class BarrierFunction(torch.nn.Module):

    def __init__(self, b_lim, power=4, barr_weight=100.0, b_buff=0.01):
        ''' Barrier Function '''
        super(BarrierFunction, self).__init__()
        self.ergodic_dim = len(b_lim)
        self.b_buff = b_buff

        if not(isinstance(power,list)):
            power = [power]*self.ergodic_dim
        self.power = torch.tensor(power).unsqueeze(1) 

        if not(isinstance(barr_weight,list)):
            barr_weight = [barr_weight]*self.ergodic_dim
        self.barr_weight = torch.tensor(barr_weight).unsqueeze(1)
        
        self.update_lims(b_lim)

    def update_lims(self,b_lim):
        self.b_lim = b_lim.clone()
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
        boundary = torch.stack([(x_check <= self.b_lim[:,0]), # lower
                                (x_check >= self.b_lim[:,1])]).T.to(int)# upper
        barr_temp = self.barr_weight*(x_check.unsqueeze(1)-self.b_lim)**self.power
        return torch.sum(boundary*barr_temp)

    def dbarr(self, x):
        dbarr_out = torch.zeros_like(x)
        x_check = x[:self.ergodic_dim]
        boundary = torch.stack([(x_check <= self.b_lim[:,0]), # lower
                                (x_check >= self.b_lim[:,1])]).T.to(int)# upper
        dbarr_temp = self.power*self.barr_weight*(x_check.unsqueeze(1)-self.b_lim)**(self.power-1)
        dbarr_out[:self.ergodic_dim] = torch.sum(boundary*dbarr_temp,1)
        return dbarr_out

    def __call__(self,x):
        return torch.stack([self.barr(xt) for xt in x])

    def __repr__(self):
        return  'BarrierFunction('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'

def dummy_conversion(x):
    return x

class TiltBarrierFunction(torch.nn.Module):

    def __init__(self, other_bar, states, tilt_lim, tilt_power=4,tilt_weight=10.0,pitch_control=False,rot_to_angles_fn=None,angles_to_rot_fn=None):
        ''' Tilt Barrier Function '''
        super(TiltBarrierFunction, self).__init__()
        self.r_idx = states.rfind('r')
        self.p_idx = states.rfind('p')
        self.b_lim = tilt_lim
        self.power = tilt_power
        self.weight = tilt_weight
        self.other_bar = other_bar

        self.w_idx = states.rfind('w')
        self.w_b_lim = other_bar.b_lim[self.w_idx].clone()

        self.rot_to_angles_fn = rot_to_angles_fn if rot_to_angles_fn is not None else dummy_conversion
        self.rpw = torch.tensor([self.r_idx,self.p_idx,self.w_idx],dtype=int)

    def update_lims(self,b_lim):
        self.other_bar.update_lims(b_lim)

    def update_ergodic_dim(self,new_ergodic_dim): 
        self.other_bar.update_ergodic_dim(new_ergodic_dim)

    def barr(self,x):
        rot = self.rot_to_angles_fn(x[self.rpw])
        tilt = torch.arccos(torch.cos(rot[0])*torch.cos(rot[1]))
        self.other_bar.b_lim[self.w_idx] = tilt/torch.pi*self.w_b_lim.clone()
        barr_temp = (tilt <= self.b_lim).to(int)*(self.weight*(tilt-self.b_lim)**self.power) # tilt <= self.b_lim
        return barr_temp + self.other_bar.barr(x)

    def dbarr(self, x):
        dbarr_temp = torch.zeros_like(x)
        rot = self.rot_to_angles_fn(x[self.rpw])
        r,p,w = rot
        tilt = torch.arccos(torch.cos(r)*torch.cos(p))
        self.other_bar.b_lim[self.w_idx] = tilt/torch.pi*self.w_b_lim.clone()
        # print(tilt)
        # if tilt <= self.b_lim:
        dbarr_temp[self.r_idx] += (tilt <= self.b_lim).to(int)*(self.power*self.weight*(tilt-self.b_lim)**(self.power - 1
                                    )*torch.sin(r)*torch.cos(p)/torch.sqrt(-torch.cos(p)**2*torch.cos(r)**2 + 1))
        dbarr_temp[self.p_idx] += (tilt <= self.b_lim).to(int)*( self.power*self.weight*(tilt-self.b_lim)**(self.power - 1
                                    )*torch.sin(p)*torch.cos(r)/torch.sqrt(-torch.cos(p)**2*torch.cos(r)**2 + 1))
        return dbarr_temp + self.other_bar.dbarr(x)

    def __call__(self,x):
        return torch.stack([self.barr(xt) for xt in x])

    def __repr__(self):
        return  'BarrierFunction('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'


class NoBarrier(torch.nn.Module):
    def __init__(self):
        pass
    def barr(self,x,others=None):
        return 0.
    def dbarr(self,x,others=None):
        return torch.zeros(len(x))
    def __call__(self,x,others=None):
        return torch.zeros(len(x))
    def __repr__(self):
        return  'NoBarrier (Dummy Function)'
    def update_ergodic_dim(self,new_ergodic_dim): 
        pass


class VelocityBarrier(torch.nn.Module):

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
            b_lim = torch.tile(torch.tensor([[-1.,1.]]),(self.state_dim,1))*b_lim
        self.b_lim = b_lim

    def barr(self,x_new,x_old):
        barr_temp = []
        b_lim_tmp = x_old.unsqueeze(1) + self.b_lim
        for i, (weight,power,b_lim,skip) in enumerate(zip(self.barr_weight,self.power,b_lim_tmp,self.skip)):
            barr_temp.append(int(not(skip))*(x_new[i] >= b_lim[1]).to(int)*(weight*(x_new[i] - (b_lim[1]))**power) ) # elif x_new[i] >= b_lim[1]:
            barr_temp.append(int(not(skip))*(x_new[i] <=  b_lim[0]).to(int)*(weight*(x_new[i] - (b_lim[0]))**power) ) # elif x_new[i] <=  b_lim[0]:
        return torch.sum(torch.stack(barr_temp))

    def dbarr(self, x_new,x_old):
        dbarr_temp = torch.zeros(len(x_new))
        b_lim_tmp = x_old.unsqueeze(1) + self.b_lim
        for i, (weight,power,b_lim,skip) in enumerate(zip(self.barr_weight,self.power,b_lim_tmp,self.skip)):
            dbarr_temp[i] += int(not(skip))*(x_new[i] >= b_lim[1]).to(int)*(power*weight*(x_new[i] - (b_lim[1]))**(power-1)) # elif x_new[i] >= b_lim[1]:            
            dbarr_temp[i] += int(not(skip))*(x_new[i] <= b_lim[0]).to(int)*(power*weight*(x_new[i] - (b_lim[0]))**(power-1)) # elif x_new[i] <=  b_lim[0]:
        return dbarr_temp

    def __call__(self,x_old,x_new):
        return torch.stack([self.barr(xt_old,xt_new) for xt_old,xt_new in zip(x_old,x_new)])

    def __repr__(self):
        return  'VelocityBarrier('+'\n\t'.join(('{} = {}'.format(item, self.__dict__[item]) for item in self.__dict__)) +')'

    def update_ergodic_dim(self,new_ergodic_dim): 
        pass
