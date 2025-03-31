#!/usr/bin/env python

### PYTHON IMPORTS ###
import numpy as np
import numpy.random as npr
from scipy.spatial.transform import Rotation
from scipy.linalg import expm

def rk4_integrate(f, dt, xt, ut):
    # rk4
    k1 = dt * f(xt,ut)
    k2 = dt * f(xt+k1/2.,ut)
    k3 = dt * f(xt+k2/2.,ut)
    k4 = dt * f(xt+k3,ut)
    return xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)


class BaseIntegratorEnv(object):

    def __init__(self,num_states,num_actions,dt,rk4,A,B,states):
        # placeholders
        self.num_states = num_states
        self.num_actions = num_actions
        self.dt = dt
        self.rk4 = rk4
        self.__A = A
        self.__B = B
        self.states = states

    def fdx(self, x, u):
        ''' Linearization wrt x '''
        return self.__A.copy()

    def fdu(self, x, u):
        ''' Linearization wrt u '''
        return self.__B.copy()

    def get_lin(self, x, u):
        ''' returns both fdx and fdu '''
        return self.fdx(x, u), self.fdu(x, u)

    def reset(self, state=None):

        if state is None:
            self.state = npr.uniform(0., .1, size=(self.num_states,))
        else:
            self.state = state[:self.num_states].copy()

        return self.state.copy()

    def f(self, x, u):
        ''' Continuous time dynamics '''
        return self.fdx(x,u) @ x + self.fdu(x,u) @ u

    def step(self, u, save=True):
        if self.rk4:
            tmp_state = rk4_integrate(self.f,self.dt,self.state,u)
        else:
            tmp_state =  self.state + self.f(self.state, u) * self.dt
        if save:
            self.state = tmp_state.copy()
        return tmp_state

class SingleIntegratorEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=np.zeros(2),states=None,rk4=True):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = dim
        A = np.zeros([self.num_states,self.num_states])
        B = np.eye(self.num_actions)
        states = states

        super(SingleIntegratorEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,A,B,states)
        self.reset(np.array(x0[:self.num_states]))

class DoubleIntegratorEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=np.zeros(4),states=None,rk4=True):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/2)
        states = states.lower()+states.upper()

        A = np.vstack([np.hstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions)*0.8]),
                              np.zeros((self.num_actions,self.num_states))])
        B = np.vstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions)])

        super(DoubleIntegratorEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,A,B,states)
        self.reset(np.array(x0))

def unhat(w_hat):
  '''
  Inputs: w_hat ~ 3x3 skew symmetric matrix
  Outputs: w ~ 3x1 matrix
  '''
  w = np.array([w_hat[2,1],w_hat[0,2],w_hat[1,0]])
  return w

def hat(w):
  '''
  Inputs: w ~ 3x1 matrix
  Outputs: w_hat ~ 3x3 skew symmetric matrix
  '''
  w_hat = np.array([[0,-w[2],w[1]],
                    [w[2],0,-w[0]],
                    [-w[1],w[0],0]])
  return w_hat

class DoubleIntegratorSpeedEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=np.zeros(6),states=None,rk4=True):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/3)
        states = states.lower()+'v'*len(states)+states.upper()

        self.A = np.vstack([np.hstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions)*0.8,np.eye(self.num_actions)*0.]),
                              np.zeros((self.num_actions*2,self.num_states))])
        self.B = np.vstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions),np.eye(self.num_actions)])

        super(DoubleIntegratorSpeedEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,self.A,self.B,states)
        self.reset(x0)

    def fdu(self, x, u):
        ''' Linearization wrt u '''
        mod = np.ones_like(x)
        signs = np.sign(x[self.num_actions:self.num_actions*2])
        signs[signs==0] = 1.
        mod[self.num_actions*2:] = signs
        return mod[:,None]*self.B.copy()

    def step(self, u, save=True):
        if self.rk4:
            tmp_state = rk4_integrate(self.f,self.dt,self.state,u)
        else:
            tmp_state =  self.state + self.f(self.state, u) * self.dt
        # force magnitude to match velocity 
        tmp_state[-self.num_actions:] = np.abs(tmp_state[self.num_actions:self.num_actions*2])
        if save:
            self.state = tmp_state.copy()
        return tmp_state
    
    def reset(self, state=None):

        if state is None:
            self.state = npr.uniform(0., .1, size=(self.num_states,))
        else:
            self.state = state[:self.num_states].copy()
            if len(self.state) < self.num_states: 
                self.state = np.hstack([self.state,np.abs(self.state[self.num_actions:self.num_actions*2])])

        return self.state.copy()

def dummy_conversion(x):
    return x

def wrap_angles(rot):
    rot[0] = rot[0] % (2 * np.pi) # wrap btwn 0 and 2*pi
    rot[1:] = (rot[1:] + np.pi) % (2 * np.pi) - np.pi # wrap btwn -pi and pi
    return rot

class DoubleIntegratorRollEnv(BaseIntegratorEnv):
    def __init__(self, dt=0.1, x0=np.zeros(12),states=None,rk4=True,rot_to_angles_fn=None,angles_to_rot_fn=None):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/2)
        self.rot_to_angles_fn = rot_to_angles_fn if rot_to_angles_fn is not None else dummy_conversion
        self.angles_to_rot_fn = angles_to_rot_fn if angles_to_rot_fn is not None else dummy_conversion

        self.__A = np.vstack([np.hstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions)*0.8]),
                              np.zeros([self.num_actions,self.num_states])])
        self.__B = np.vstack([np.zeros((self.num_actions,self.num_actions)),np.eye(self.num_actions)])

        rot = np.ones(3,dtype=int)*-1 #rpw
        tmp_states = ''
        for idx,key in enumerate(states):
            if key == 'r':
                rot[0] = idx
            elif key == 'p':
                rot[1] = idx
            elif key == 'w':
                rot[2] = idx
            else: 
                tmp_states = tmp_states + key
        assert not(any(rot<0)),f'need roll, pitch, and yaw to use this dynamics model, got states {states}'
        self.rpw = rot
        self.d_rpw = rot+self.num_actions
        tmp_states = tmp_states + 'rpw'
        states = tmp_states.lower()+tmp_states.upper()

        super(DoubleIntegratorRollEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,self.__A,self.__B, states)

        self.rot_idxs = np.array([[idx_r,idx_c] for idx_r in self.rpw for idx_c in self.d_rpw])

        self.reset(np.array(x0))

    def get_new_rot(self,x):
        rot = self.rot_to_angles_fn(x[self.rpw])
        w = x[self.d_rpw]
        R = Rotation.from_euler('xyz',rot).as_matrix()
        Rnew = expm(hat(w)*self.dt) @ R  # A(t) = e^(W*dt)*A(t-1)
        new_rot = Rotation.from_matrix(Rnew).as_euler('xyz')
        new_rot = wrap_angles(new_rot)
        return self.angles_to_rot_fn(new_rot)
        
    # def rk4_integrate_w(self,w):
    #     # rk4
    #     k1 = self.dt*w
    #     k2 = self.dt*(w + k1/2)
    #     k3 = self.dt*(w + k2/2)
    #     k4 = self.dt*(w + k3)
    #     return (1/6.) * (k1+2.0*k2+2.0*k3+k4)

    # def get_drot(self,x):
    #     w = x[self.d_rpw]
    #     cayley = (np.eye(3)+0.5*hat(w))*np.linalg.inv(np.eye(3)-0.5*hat(w)) # infinitesimal rotation matrix
    #     return Rotation.from_matrix(cayley).as_euler('xyz')

    def fdx(self, x, u):
        ''' Linearization wrt x '''
        tmp_A = self.__A.copy()
        rot = self.rot_to_angles_fn(x[self.rpw])
        r,p,w = rot
        R = Rotation.from_euler('xyz',rot).as_matrix()
        B = np.array([[1., np.sin(r)*np.tan(p),np.cos(r)*np.tan(p)],
                      [0., np.cos(r),         -np.sin(r)],
                      [0., np.sin(r)/np.cos(p),np.cos(r)/np.cos(p)]])
        B = B @ R
        for val_r, idx_r in enumerate(self.rpw):
            for val_c, idx_c in enumerate(self.d_rpw):
                tmp_A[idx_r,idx_c] = B[val_r,val_c]
        return tmp_A

    def step(self, u, save=True):
        if self.rk4:
            tmp_state = rk4_integrate(self.f,self.dt,self.state,u)
        else:
            tmp_state =  self.state + self.f(self.state, u) * self.dt
        # override rotation
        tmp_state[self.rpw] = self.get_new_rot(self.state)
        # print(tmp_state[self.rpw],tmp_state[self.d_rpw])
        if save:
            self.state = tmp_state.copy()
        return tmp_state

    def reset(self, state=None):
        if state is None:
            self.state = npr.uniform(0., .1, size=(self.num_states,))
        else:
            # then update
            self.state = state[:self.num_states].copy()

        return self.state.copy()
