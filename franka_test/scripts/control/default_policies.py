#!/usr/bin/env python
import numpy as np

##### default policies
class Roll(object):
    def __init__(self,model,horizon):
        self.num_actions = model.num_actions
        self._dx = np.zeros([model.num_actions,model.num_states])
        self.u = iter([])

    def reset(self,x=None,u=None,iter_idx=0):
        if iter_idx < 0:
            u = np.roll(u,iter_idx,0)
            u[iter_idx:] = 0.
        self.u = iter(u)
        return u

    def dx(self,x=None,u=None):
        return self._dx.copy()

    def __call__(self,x=None):
        try:
            return next(self.u)
        except StopIteration:
            print('out of controls')
            return np.zeros(self.num_actions)

class Zero(object):
    def __init__(self,model,horizon):
        self.num_actions = model.num_actions
        self._dx = np.zeros([model.num_actions,model.num_states])
        self.u = iter([])

    def reset(self,x=None,u=None,iter_idx=0):
        if iter_idx < 0:
            u = np.zeros(u.shape)
        self.u = iter(u)
        return u

    def dx(self,x=None,u=None):
        return self._dx.copy()

    def __call__(self,x=None):
        try:
            return next(self.u)
        except StopIteration:
            return np.zeros(self.num_actions)

class LQR(object):
    def __init__(self,model,horizon):
        from scipy.linalg import solve_continuous_are
        A,B =  model.get_lin(np.ones(model.num_states),np.ones(model.num_actions))
        Q = np.diag([5.]*model.num_actions+[1.]*model.num_actions)
        R = np.eye(model.num_actions)*100.*horizon
        P = solve_continuous_are(A,B,Q,R,balanced=False)
        self.Klqr = np.linalg.inv(R) @ B.T @ P

    def reset(self,x=None,u=None,iter_idx=0):
        return u

    def dx(self,x=None,u=None):
        return -self.Klqr.copy()

    def __call__(self,x):
        return - self.Klqr @ x
