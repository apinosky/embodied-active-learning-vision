#!/usr/bin/env python
import torch

##### default policies
class Roll(torch.nn.Module):
    def __init__(self,model,horizon):
        super(Roll, self).__init__()
        self.num_actions = model.num_actions
        self.dtype = model.dtype
        self._dx = torch.zeros([model.num_actions,model.num_states],dtype=self.dtype)
        self.u = iter([])

    def reset(self,x=None,u=None,iter_idx=0):
        if iter_idx < 0:
            u = torch.roll(u,iter_idx,0)
            u[iter_idx:] = 0.
        self.u = iter(u)
        return u

    def dx(self,x=None,u=None):
        return self._dx.clone()

    def __call__(self,x=None):
        try:
            return next(self.u)
        except StopIteration:
            print('out of controls')
            return torch.zeros(self.num_actions,dtype=self.dtype)

class Zero(torch.nn.Module):
    def __init__(self,model,horizon):
        super(Zero, self).__init__()
        self.num_actions = model.num_actions
        self.dtype = model.dtype        
        self._dx = torch.zeros([model.num_actions,model.num_states],dtype=self.dtype)
        self.u = iter([])

    def reset(self,x=None,u=None,iter_idx=0):
        if iter_idx < 0:
            u = torch.zeros_like(u)
        self.u = iter(u)
        return u

    def dx(self,x=None,u=None):
        return self._dx.clone()

    def __call__(self,x=None):
        try:
            return next(self.u)
        except StopIteration:
            return torch.zeros(self.num_actions,dtype=self.dtype)

class BarrierPush(torch.nn.Module):
    def __init__(self,model,horizon):
        super(BarrierPush, self).__init__()
        self.num_actions = model.num_actions
        self.dtype = model.dtype
        self._dx = torch.zeros([model.num_actions,model.num_states],dtype=self.dtype)
        self.u = iter([])
        self.b_lim = [[-1.,1.]]*model.num_states
        self.skip = [True if s.upper() == s else False for s in model.states] # only use states
        self.weight = 5.

    def reset(self,x=None,u=None,iter_idx=0):
        if iter_idx > 0:
            self.u = iter(u)
        else:
            self.u = iter([])
        return u

    def clipped(self,x,u):
        for i, (b_lim,skip) in enumerate(zip(self.b_lim,self.skip)):
            if skip:
                pass
            else: 
                if (((x[i] >= b_lim[1]) and (x[i + self.num_actions] > 0)) or 
                    ((x[i] <=  b_lim[0]) and (x[i + self.num_actions] < 0))):
                    u[i] = -self.weight*x[i + self.num_actions]
        return u

    def dx(self,x=None,u=None):
        dx = self._dx.clone()
        for i, (b_lim,skip) in enumerate(zip(self.b_lim,self.skip)):
            if skip:
                pass
            else: 
                if (((x[i] >= b_lim[1]) and (x[i + self.num_actions] > 0)) or 
                    ((x[i] <=  b_lim[0]) and (x[i + self.num_actions] < 0))):
                    dx[i,i+self.num_actions] = -self.weight
        return dx

    def __call__(self,x=None):
        try:
            u = next(self.u)
        except StopIteration:
            u = torch.zeros(self.num_actions,dtype=self.dtype)
        return self.clipped(x,u)

import numpy as np
class LQR(torch.nn.Module):
    def __init__(self,model,horizon):
        super(LQR, self).__init__()
        from scipy.linalg import solve_continuous_are
        A,B =  model.get_lin(torch.ones(model.num_states),torch.ones(model.num_actions))
        A = A.numpy()
        B = B.numpy()
        Q = np.diag([5.]*model.num_actions+[1.]*model.num_actions)
        R = np.eye(model.num_actions)*100.*horizon
        P = solve_continuous_are(A,B,Q,R,balanced=False)
        self.Klqr = torch.as_tensor(np.linalg.inv(R) @ B.T @ P,dtype=model.dtype)

    def reset(self,x=None,u=None,iter_idx=0):
        return u

    def dx(self,x=None,u=None):
        return -self.Klqr.clone()

    def __call__(self,x):
        return - self.Klqr @ x
