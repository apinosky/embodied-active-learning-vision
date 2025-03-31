#!/usr/bin/env python

### PYTHON IMPORTS ###
import torch
from .rotations import euler_angles_to_matrix, matrix_to_euler_angles

def rk4_integrate(f, dt, xt, ut):
    # rk4
    k1 = dt * f(xt,ut)
    k2 = dt * f(xt+k1/2.,ut)
    k3 = dt * f(xt+k2/2.,ut)
    k4 = dt * f(xt+k3,ut)
    return xt + (1/6.) * (k1+2.0*k2+2.0*k3+k4)


class BaseIntegratorEnv(torch.nn.Module):

    def __init__(self,num_states,num_actions,dt,rk4,A,B,states,dtype=torch.float32):
        super(BaseIntegratorEnv, self).__init__()
        # placeholders
        self.num_states = num_states
        self.num_actions = num_actions
        self.dt = torch.tensor(dt,dtype=dtype)
        self.rk4 = rk4
        self.__A = A
        self.__B = B
        self.states = states
        self.state_dist = torch.distributions.Uniform(0., .1)
        self.dtype = dtype

    def fdx(self, x, u):
        ''' Linearization wrt x '''
        return self.__A.clone()

    def fdu(self, x, u):
        ''' Linearization wrt u '''
        return self.__B.clone()

    def get_lin(self, x, u):
        ''' returns both fdx and fdu '''
        return self.fdx(x, u), self.fdu(x, u)

    def reset(self, state=None):
        if state is None:
            self.state = self.state_dist.sample((self.num_states,))
        else:
            if isinstance(state,torch.Tensor): 
                self.state = state[:self.num_states].clone()
            else:
                self.state = torch.tensor(state[:self.num_states],dtype=self.dtype)

        return self.state.clone()

    def f(self, x, u):
        ''' Continuous time dynamics '''
        return self.fdx(x,u) @ x + self.fdu(x,u) @ u

    def step(self, u, save=True):
        if self.rk4:
            tmp_state = rk4_integrate(self.f,self.dt,self.state,u)
        else:
            tmp_state =  self.state + self.f(self.state, u) * self.dt
        if save:
            self.state = tmp_state.clone()
        return tmp_state

class SingleIntegratorEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=torch.zeros(2),states=None,rk4=True,dtype=torch.float32):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = dim
        A = torch.zeros([self.num_states,self.num_states])
        B = torch.eye(self.num_actions)
        states = states

        super(SingleIntegratorEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,A,B,states,dtype=dtype)
        self.reset(x0)

class DoubleIntegratorEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=torch.zeros(4),states=None,rk4=True,dtype=torch.float32):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/2)
        states = states.lower()+states.upper()

        A = torch.vstack([torch.hstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions)*0.8]),
                              torch.zeros((self.num_actions,self.num_states))])
        B = torch.vstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions)])

        super(DoubleIntegratorEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,A,B,states,dtype=dtype)
        self.reset(x0)

class DoubleIntegratorSpeedEnv(BaseIntegratorEnv):

    def __init__(self, dt=0.1, x0=torch.zeros(6),states=None,rk4=True,dtype=torch.float32):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/3)
        states = states.lower()+'v'*len(states)+states.upper()

        self.A = torch.vstack([torch.hstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions)*0.8,torch.eye(self.num_actions)*0.]),
                              torch.zeros((self.num_actions*2,self.num_states))])
        self.B = torch.vstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions),torch.eye(self.num_actions)])

        super(DoubleIntegratorSpeedEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,self.A,self.B,states,dtype=dtype)
        self.reset(x0)

    def fdu(self, x, u):
        ''' Linearization wrt u '''
        mod = torch.ones_like(x)
        signs = x[self.num_actions:self.num_actions*2].sign()
        signs[signs==0] = 1.
        mod[self.num_actions*2:] = signs
        return mod.unsqueeze(1)*self.B.clone()

    def step(self, u, save=True):
        if self.rk4:
            tmp_state = rk4_integrate(self.f,self.dt,self.state,u)
        else:
            tmp_state =  self.state + self.f(self.state, u) * self.dt
        # force magnitude to match velocity 
        tmp_state[-self.num_actions:] = tmp_state[self.num_actions:self.num_actions*2].abs()
        if save:
            self.state = tmp_state.clone()
        return tmp_state

    def reset(self, state=None):
        if state is None:
            self.state = self.state_dist.sample((self.num_states,))
        else:
            if isinstance(state,torch.Tensor): 
                self.state = state[:self.num_states].clone()
            else:
                self.state = torch.tensor(state[:self.num_states],dtype=self.dtype)
            if len(self.state) < self.num_states: 
                self.state = torch.hstack([self.state,torch.abs(self.state[self.num_actions:self.num_actions*2])])
        return self.state.clone()


# class SemiDoubleIntegratorEnv(BaseIntegratorEnv):

#     def __init__(self, dt=0.1, x0=torch.zeros(4),states=None,rk4=True,dtype=torch.float32):

#         dim = len(x0)
#         self.num_states = dim
#         self.num_actions = dim
#         num_accel = int(dim/2)
#         states = states.lower()+states.upper()

#         A = torch.vstack([torch.hstack([torch.zeros((num_accel,num_accel)),torch.eye(num_accel)*0.8]),
#                               torch.zeros((num_accel,self.num_states))])
#         B = torch.eye(self.num_actions)
#         super(SemiDoubleIntegratorEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,A,B,states,dtype=dtype)
#         self.reset(x0)

def dummy_conversion(x):
    return x

def unhat(w_hat):
    '''
    Inputs: w_hat ~ 3x3 skew symmetric matrix
    Outputs: w ~ 3x1 matrix
    '''
    locs = torch.tensor([[2,1],[0,2],[1,0]])
    return w_hat[locs[:,0],locs[:,1]]

def hat(w):
    '''
    Inputs: w ~ 3x1 matrix
    Outputs: w_hat ~ 3x3 skew symmetric matrix
    '''
    w_hat = torch.zeros(3,3,dtype=w.dtype)
    w_hat[0,1] = -w[2]
    w_hat[0,2] = w[1]
    w_hat[1,0] = w[2]
    w_hat[1,2] = -w[0]
    w_hat[2,0] = -w[1]
    w_hat[2,1] = w[0]
    # w_hat = torch.tensor([[0,-w[2],w[1]],
    #                     [w[2],0,-w[0]],
    #                     [-w[1],w[0],0]])
    return w_hat

def get_B(rot,R):
    # rot = [roll,pitch,yaw]

    ## prevent singularity @ pitch = pi/2
    # if torch.abs(rot[1] - torch.pi/2) < 1e-5: 
    rot[1] += 1e-5
    
    # B = torch.tensor([[1, torch.sin(rot[0])*torch.tan(rot[1]),torch.cos(rot[0])*torch.tan(rot[1])],
    #                     [0, torch.cos(rot[0]),                 -torch.sin(rot[0])],
    #                     [0, torch.sin(rot[0])/torch.cos(rot[1]),torch.cos(rot[0])/torch.cos(rot[1])]])
    B = torch.eye(3,dtype=rot.dtype)
    s0 = torch.sin(rot[0])
    c0 = torch.cos(rot[0])
    t1 = torch.tan(rot[1])
    c1 = torch.cos(rot[1])
    B[0,1] =  s0*t1
    B[0,2] =  c0*t1
    B[1,1] =  c0
    B[1,2] = -s0
    B[2,1] =  s0/c1
    B[2,2] =  c0/c1
    B = B @ R
    return B.ravel()

def get_new_rot(R,d_rot,dt):
    w_hat = hat(d_rot)
    Rnew = torch.matrix_exp(w_hat*dt) @ R  # A(t) = e^(W*dt)*A(t-1)
    new_rot = wrap_angles(matrix_to_euler_angles(Rnew,'XYZ') )
    return Rnew, new_rot

def wrap_angles(rot):
    rot[0] = rot[0] % (2 * torch.pi) # wrap btwn 0 and 2*pi
    rot[1:] = (rot[1:] + torch.pi) % (2 * torch.pi) - torch.pi # wrap btwn -pi and pi
    return rot

class DoubleIntegratorRollEnv(BaseIntegratorEnv):
    def __init__(self, dt=0.1, x0=torch.zeros(12),states=None,rk4=True,dtype=torch.float32,rot_to_angles_fn=None,angles_to_rot_fn=None):

        dim = len(x0)
        self.num_states = dim
        self.num_actions = int(dim/2)

        self.__A = torch.vstack([torch.hstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions)*0.8]),
                              torch.zeros([self.num_actions,self.num_states])]).to(dtype=dtype)
        self.__B = torch.vstack([torch.zeros((self.num_actions,self.num_actions)),torch.eye(self.num_actions)]).to(dtype=dtype)

        rot = torch.ones(3,dtype=int)*-1 #rpw
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

        super(DoubleIntegratorRollEnv, self).__init__(self.num_states,self.num_actions,dt,rk4,self.__A,self.__B, states,dtype=dtype)

        ## jit script
        if rot_to_angles_fn is not None:
            self.rot_to_angles_fn = torch.jit.script(rot_to_angles_fn.eval())
            self.rot_to_angles_fn = torch.jit.freeze(self.rot_to_angles_fn)
        else: 
            self.rot_to_angles_fn = dummy_conversion
        if angles_to_rot_fn is not None:
            self.angles_to_rot_fn = torch.jit.script(angles_to_rot_fn.eval())
            self.angles_to_rot_fn = torch.jit.freeze(self.angles_to_rot_fn)
        else: 
            self.angles_to_rot_fn = dummy_conversion

        self.get_B_jit = torch.jit.script(get_B)
        self.get_new_rot_jit = torch.jit.script(get_new_rot)
        self.R = torch.eye(3)
        
        self.rot_idxs = torch.tensor([[idx_r,idx_c] for idx_r in self.rpw for idx_c in self.d_rpw])

        self.reset(x0)

    def get_new_rot(self,x):
        self.R, new_rot = self.get_new_rot_jit(self.R,x[self.d_rpw],self.dt)
        return self.angles_to_rot_fn(new_rot)

    # def get_drot(self,x):
    #     w = x[self.d_rpw]
    #     cayley = (torch.eye(3,dtype=self.dtype)+0.5*hat(w))*torch.linalg.inv(torch.eye(3,dtype=self.dtype)-0.5*hat(w)) # infinitesimal rotation matrix
    #     return matrix_to_euler_angles(cayley,'ZYX') 

    def fdx(self, x, u):
        ''' Linearization wrt x '''
        tmp_A = self.__A.clone()
        rot = self.rot_to_angles_fn(self.state[self.rpw])
        B = self.get_B_jit(rot,self.R)
        tmp_A[self.rot_idxs[:,0],self.rot_idxs[:,1]] = B
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
            self.state = tmp_state.clone()
        return tmp_state

    def reset(self, state=None):
        if state is None:
            self.state = self.state_dist.sample((self.num_states,))
        else:
            if isinstance(state,torch.Tensor): 
                self.state = state[:self.num_states]
            else:
                self.state = torch.tensor(state[:self.num_states],dtype=self.dtype)
        rot = self.rot_to_angles_fn(self.state[self.rpw])
        self.R = euler_angles_to_matrix(rot,'XYZ') # equivalent to scipy.spatial.transform.Rotation.from_euler('xyz',rot).as_matrix()


        return self.state.clone()

