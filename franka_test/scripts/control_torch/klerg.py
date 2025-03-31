#!/usr/bin/env python

import numpy as np
import torch

import itertools
from termcolor import cprint
import yaml
import time 

from .memory_buffer import MemoryBuffer_torch, AvoidDist
from .dynamics import rk4_integrate
from .barrier import *
from .default_policies import *
from .klerg_utils import *

import sys, os
base_path = os.path.dirname(os.path.abspath(__file__))
from franka.franka_utils import ws_conversion, find_non_vel_locs

use_fut = False
single = True

def rescale(x,old_minmax,new_minmax):
    return (x - old_minmax[0])/(old_minmax[1]-old_minmax[0])*(new_minmax[1]-new_minmax[0]) + new_minmax[0]

class PriorDist():
    def __init__(self,states): 
        base_states = 'xyzrpw'
        # base_duck = [-0.8,-0.8,-0.15,3.1,0.,0.]
        # base_ball = [0.6,0.9,-0.15,3.1,0.,0.]
        # base_covar = [0.2,0.2,0.5,10.,10.,10.]

        base_duck = [-0.8,-0.8,-0.15,3.6,0.5,0.]
        base_ball = [0.6,0.9,-0.15,2.6,-0.5,0.]
        base_covar = [0.2,0.2,0.5,0.2,0.2,0.5]

        duck = [base_duck[base_states.rfind(s)] if (s in base_states) else 0. for s in states]
        ball = [base_ball[base_states.rfind(s)] if (s in base_states) else 0. for s in states]
        covar = [base_covar[base_states.rfind(s)] if (s in base_states) else 1. for s in states]

        locs = [torch.FloatTensor(duck),torch.FloatTensor(ball)]
        covar = torch.diag(torch.FloatTensor(covar))
        self.tdists = [torch.distributions.MultivariateNormal(loc,covar) for loc in locs]
    def pdf(self, x):
        x = torch.as_tensor(x) 
        pdf = self.pdf_torch(x)
        return pdf.numpy()
    def pdf_torch(self,samples):
        return torch.sum(torch.stack([tdist.log_prob(samples).exp() for tdist in self.tdists]),0)+1e-5


class dummyTestDist():
    def __init__(self,center=None,covar=None): 
        self.dtype=torch.float32
        if center == None: 
            center = torch.FloatTensor([1.2,1.0])
        if covar == None: 
            covar = torch.eye(len(center)).to(self.dtype)
        self.tdist = torch.distributions.MultivariateNormal(center,covar)
    def pdf(self, x):
        x = torch.as_tensor(x) 
        pdf = self.pdf_torch(x)
        return pdf.numpy()
    def pdf_torch(self,samples):
        return self.tdist.log_prob(samples).exp()

class normalEnv():
    def __init__(self,lims,covar=None): 
        self.dtype=torch.float32
        self.lims = lims
        center = torch.mean(lims,1)
        if covar == None: 
            covar = torch.eye(len(center)).to(self.dtype)
        self.tdist = torch.distributions.MultivariateNormal(center,covar)
    def pdf(self, x):
        x = torch.as_tensor(x) 
        pdf = self.pdf_torch(x)
        return pdf.numpy()
    def pdf_torch(self,samples):
        return self.tdist.log_prob(samples).exp()

debug = False

class Robot(object):
    """ Robot class that runs the KL-Erg MPC Planner """
    def __init__(self, x0, robot_lim, explr_idx, explr_robot_lim_scale = 1.0, target_dist=None, dt=0.1,
                    R=0.01, use_vel = True,pybullet=False,
                    horizon=10, buffer_capacity=100, std=0.05, std_plot=0.05, plot_data=False, plot_extra=False,
                    states='xy',plot_states='xy',tray_lim=None,robot_ctrl_lim=None,
                    uniform_tdist=False, vel_states=False, use_magnitude=False):

        self.load_yaml(uniform_tdist) 

        if debug: 
            self.target_dist=dummyTestDist()
        else:
            self.target_dist=target_dist
        for attr,val in zip(['dtype','device'],[torch.float32,'cpu']):
            if not hasattr(self.target_dist,attr):
                setattr(self.target_dist,attr,val)
        self.dtype=self.target_dist.dtype 
        self.device='cpu'
        torch.set_default_dtype(self.dtype)
        self.traj_footprint_vec = traj_footprint_vec
        self.kldiv_grad_vec = kldiv_grad_vec

        self.prior_dist = PriorDist(states) 
        self.use_prior = False 
        self.average_prior = False
        self.pybullet = pybullet

        # Set up Workspace
        self.robot_lim = torch.tensor(robot_lim,dtype=self.dtype)
        self.explr_idx = torch.tensor(explr_idx) # these are the indexes based on inputs states
        self.states = states
        self.robot_ctrl_lim = robot_ctrl_lim
        self.uniform_tdist = uniform_tdist
        self.vel_states = vel_states
        self.horizon = horizon
        self.plot_extra = plot_extra
        self.plot_smooth = True
        self.use_magnitude = use_magnitude
        self.use_vel = use_vel
        if tray_lim is not None: 
            self.tray_lim = torch.tensor(tray_lim,dtype=self.dtype)
        else: 
            self.tray_lim = tray_lim

        if len(states) == len(plot_states): 
           self.plot_extra = False 
           self.plot_smooth = False 

        ## modify inputs to accomodate double integrator
        if self.vel_states: # vel already accounted for in double integrator
            self.non_vel_locs,self.vel_locs, states = find_non_vel_locs(self.states)
            x0 = np.hstack([np.array(x0)[self.non_vel_locs],np.zeros(len(self.non_vel_locs))])
        else: 
            self.non_vel_locs = list(range(len(self.states)))
            self.use_magnitude = False

        extra_args = {}
        if sum([rot in self.states for rot in 'rpw']) > 1: # if more than one angle
            self.rot_states = True
            rpw = [idx for idx,key in enumerate(self.states) if key in 'rpw' ]
            # assert torch.all(self.robot_lim[rpw] == self.tray_lim[rpw]),'robot_lim needs to use real angles for roll,pitch,yaw to use current DoubleIntegratorRollEnv implementation'
            if not(torch.all(self.robot_lim[rpw] == self.tray_lim[rpw])):
                extra_args['rot_to_angles_fn'] = Lambda(ws_conversion,(self.robot_lim[rpw],self.tray_lim[rpw]))
                extra_args['angles_to_rot_fn'] = Lambda(ws_conversion,(self.tray_lim[rpw],self.robot_lim[rpw]))
            from .dynamics import DoubleIntegratorRollEnv as dynamics
        else:
            self.rot_states = False
            if self.use_magnitude:
                from .dynamics import DoubleIntegratorSpeedEnv as dynamics
                x0 = np.hstack([x0,np.zeros(len(self.non_vel_locs))])
            else:
                from .dynamics import DoubleIntegratorEnv as dynamics

        # Set up robot & planner
        dt_scale = 1. if self.use_vel else 3.
        self.robot = dynamics(dt=dt*dt_scale,x0=x0,states=states,dtype=self.dtype,**extra_args)
        self.explr_locs = torch.tensor([idx for idx,s in enumerate(self.robot.states) if s in self.states]) # these are the indexes based on dynamics
        self.planner = dynamics(dt=dt,x0=x0,states=states,dtype=self.dtype,**extra_args)

        if 'b' in self.robot.states:
            self.bv_idx = self.robot.states.rfind('B')

        # Set up sampling
        self.lims = self.robot_lim.clone()
        self.lims += torch.tile(torch.tensor([[-1.,1.]]),(len(self.lims),1))*(self.lims[:,[1]]-self.lims[:,[0]])*(explr_robot_lim_scale-1.)/2.      
        if self.use_magnitude:
            self.lims[self.vel_locs,0] = 0.
        self.env_sampler =  torch.distributions.Uniform(*self.lims[self.explr_idx].T)
        if self.optimize_samples: 
            if self.vel_states: 
                var = torch.tensor([1. if state==state.lower() else 2. for state in self.states],dtype=self.dtype)
                kernel_covar = torch.diag(var*0.001)
            else:
                kernel_covar = torch.diag(torch.ones(len(self.explr_idx))*0.001)
            self.kernel_dist = torch.distributions.MultivariateNormal(torch.zeros(len(self.explr_idx)),kernel_covar)
        if self.sample_near_current_loc: 
            self.loc_sampler = torch.distributions.Normal(torch.zeros_like(self.std),self.std*4.)

        # Set up other parameters
        self.num_iters_per_step = max(1,int(self.pct_of_horizon_for_inner_loop*self.horizon))
        # self.std = torch.ones(len(self.states),dtype=self.dtype)*std
        self.std = torch.tensor([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std
        self.std_plot = torch.tensor([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std_plot
        # self.std = torch.tensor([1. if state in 'xyzb' else (2. if state in 'rpw' else 5.) for state in self.states],dtype=self.dtype)*std
        # self.std = torch.tensor([1. if state in 'xyb' else (0.5 if state in 'rpwz' else 5.) for state in self.states],dtype=self.dtype)*std
        if isinstance(R,(int,float)):
            R = [R]*self.robot.num_actions
        self.R_inv = torch.inverse(torch.diag(torch.tensor(R,dtype=self.dtype)))
        self.u = torch.zeros((self.horizon, self.planner.num_actions),dtype=self.dtype)
        self.memory_buffer = MemoryBuffer_torch(buffer_capacity,self.planner.num_states,dtype=self.dtype)
        # self.control_lim = torch.tensor([[-1.0,1.0] for state in states],dtype=self.dtype)
        self.control_lim = torch.tensor([[-0.5,0.5] if state in 'z' else [-1.0,1.0] for state in states],dtype=self.dtype)

        # set up default policy
        DefaultPolicy = eval(self.DefaultPolicy)
        self.policy = DefaultPolicy(self.planner,self.horizon)

        # other setup
        self.plot_data = plot_data
        self.plot_states = plot_states
        self.barrier,self.barr_lim = setup_barrier(states,self.robot_lim,self.robot_ctrl_lim,self.non_vel_locs,self.dtype,extra_args,self.rot_states,uniform=uniform_tdist)
        if self.full_cost and not self.rot_states:
            self.batched_cost = torch.func.vmap(self.get_cost)

        self.count = 0
        cprint('[klerg] setup complete','cyan')
        
    def load_yaml(self,uniform):
        if uniform: 
            file =  '/robot_config_uniform.yaml'
        else: 
            file = '/robot_config.yaml'
        with open(base_path+file) as f:
            yaml_config = yaml.load(f,Loader=yaml.FullLoader)
        for k, v in yaml_config.items():
            setattr(self, k, v)

    def setup_plotting(self,num_samples=100): 
        if self.plot_data:
            # self.plot_data = [0]*6 # placeholder for [samples tdist edist planned_traj]
            state = self.robot.state.clone()
            num_samples += 4
            samples = self.env_sampler.sample((num_samples,))
            dummy_qp = renormalize(torch.ones(num_samples))
            dummy_locs = torch.tile(state[self.explr_locs].unsqueeze(0),(self.horizon+1,1))
            dummy_cost = torch.tensor([1000.])
            self.plot_data = [samples] + [dummy_qp]*2 + [dummy_locs] + [dummy_qp]*2 + [dummy_cost]

            # get all combinations of exploration states
            self.all_plot_states = [x[0]+x[1] for x in itertools.combinations(self.states,2)]
            self.all_plot_idx = [torch.tensor([self.states.rfind(s) for s in ps]) for ps in self.all_plot_states] # these are the indexes based on inputs states
            if any(torch.hstack(self.all_plot_idx)==-1):
                raise ValueError('robot controller (klerg) did not find requested plot state')
            self.all_corner_samples = [self.get_corners(ps) for ps in self.all_plot_idx]
            
            # main plot states
            self.desired_plot_idx = np.argwhere(np.array(self.all_plot_states) == self.plot_states).item()
            self.plot_idx = self.all_plot_idx[self.desired_plot_idx]
            self.corner_samples = self.all_corner_samples[self.desired_plot_idx]
            self.corners = torch.ones(len(self.corner_samples),dtype=self.dtype)
        else:
            self.plot_idx = torch.tensor([self.states.rfind(s) for s in self.plot_states]) 
            self.plot_data = None
            self.test_corners = False
        self.last_plan = torch.vstack([self.robot.state]+[self.robot.step(ut) for ut in self.u])

    @torch.no_grad()
    def update_lims(self,idx,lims): 
        if not(isinstance(lims,torch.Tensor)):
            lims = torch.tensor(lims,dtype=self.dtype)
        self.lims[idx] = lims
        if self.use_magnitude:
            self.lims[self.vel_locs,0] = 0.
        self.env_sampler =  torch.distributions.Uniform(*self.lims[self.explr_idx].T)
        self.update_corners()
        if self.use_barrier:
            barr_lim = torch.tensor(self.lims[self.non_vel_locs].tolist() + self.robot_ctrl_lim.tolist(),dtype=self.dtype) # for barrier function
            self.barrier.update_lims(barr_lim)

    @torch.no_grad()
    def update_corners(self):
        self.corner_samples = self.get_corners(self.plot_idx)

    @torch.no_grad()
    def get_corners(self,plot_idx):
        corner_samples = torch.tensor(list(itertools.product(*self.lims[[plot_idx]])),dtype=self.dtype)
        if len(self.explr_idx) > 2:
            corner_dim = len(self.explr_idx)
            tmp_corner_samples = torch.zeros((corner_samples.shape[0],corner_dim),dtype=self.dtype)
            tmp_corner_samples[:,plot_idx] = corner_samples
            corner_samples = tmp_corner_samples
        return corner_samples

    def step(self, num_target_samples= 50, num_traj_samples=30,save_update=False, temp=1.0):
        # temp = len(self.states)
        self.kldiv_planner(num_target_samples= num_target_samples, num_traj_samples= num_traj_samples, temp=temp)

        ctrl = self.u[0].clone()
        if not save_update:
            state = self.robot.step(ctrl,save=False)
        else:
            state = self.robot.step(ctrl)
            self.save_update(state,save=True)
        # state = self.saturate_vel(state)
        vel = state[self.planner.num_actions:]
        return state[self.explr_locs].numpy(), vel.numpy(), ctrl.numpy()

    @torch.no_grad()
    def save_update(self,full_state,force=0.,save=True):
        if not isinstance(full_state,torch.Tensor): 
            full_state = torch.tensor(full_state,dtype=self.dtype)
        if torch.any(torch.isnan(full_state)): # don't save update if you got a nan
            print('got nan in full_state')
            return 
        if 'b' in self.states: 
            full_state[self.bv_idx] = self.last_plan[1][self.bv_idx] # not provided by ROS (always = 0)

        ''' find closest loc '''
        if self.pybullet:
            policy_idx = torch.norm(self.last_plan[:,self.non_vel_locs] - full_state[self.non_vel_locs],dim=1).argmin().item() # which is closest
        else: 
            policy_idx = torch.norm(self.last_plan - full_state,dim=1).argmin().item() # which is closest
        planned_state = self.last_plan[policy_idx]

        ''' update state '''
        if self.pybullet:
            vel_smoothing = 0.5
        else:
            vel_smoothing = 0.8
        full_state[self.planner.num_actions:] = vel_smoothing*full_state[self.planner.num_actions:] + (1-vel_smoothing)*planned_state[self.planner.num_actions:]
        x = self.robot.reset(full_state)
        
        ''' update control based on state '''
        self.u = self.policy.reset(x,self.u.clone(),-policy_idx)
        # self.last_plan = self.last_plan[policy_idx:]

        if save:
            self.memory_buffer.push(x.clone())


    @torch.no_grad()
    def test(self,num_target_samples=100,N=10):
        N = torch.as_tensor(N)
        traj_samples = torch.randn(N+self.horizon,self.robot.num_states)
        x = traj_samples[0]
        samples = self.env_sampler.sample((num_target_samples,))
        importance_ratio = self.env_sampler.sample((num_target_samples,)).sum(1)
        # jit
        self.traj_footprint_vec_jit = torch.jit.trace(traj_footprint_vec,(traj_samples,samples,self.explr_locs,self.std,N))
        self.kldiv_grad_vec = torch.jit.trace(kldiv_grad_vec,(x, samples, self.explr_locs, self.std, importance_ratio,N))
        # warmup
        self.traj_footprint_vec_jit(traj_samples,samples,self.explr_locs,self.std,N)
        self.kldiv_grad_vec(x, samples, self.explr_locs, self.std, importance_ratio,N)
        # plotting 
        self.setup_plotting(num_target_samples)

    @torch.no_grad()
    def saturate_control(self,u,app_thresh=0.1):
        # r = torch.arange(len(self.control_lim))
        # c = (u > 0).to(int)
        # out = self.control_lim[r,c].clone()
        # out[torch.abs(u) < app_thresh] = 0.
        out = torch.tanh(u/app_thresh)*self.control_lim[:,1]
        return out

    def saturate_vel(self,x,app_thresh=0.2):
        vel = x[self.planner.num_actions:] # still attached so updating x
        h = (vel > 0)
        vel[h] = torch.clamp(vel[h],app_thresh,None)
        l = (vel < 0)
        vel[l] = torch.clamp(vel[l],None,-app_thresh)
        return x
    
    # Define kernel
    def kernel(self,x1, x2):
        return self.kernel_dist.log_prob(x1-x2).exp()

    def kernel_loss(self,pts): 
        inner_prod = torch.mean(self.kernel(pts.unsqueeze(1),pts.unsqueeze(0)))
        return inner_prod - 2 * torch.mean(renormalize(self.target_dist.pdf_torch(pts))) # + torch.mean(self.barrier(pts))

    def get_samples(self,num_target_samples,num_traj_samples):
        # Sample Target Distribution
        if self.add_recent_history:
            recent = self.memory_buffer.get_recent(self.horizon)
            num_target_samples -= len(recent)
        if self.sample_near_current_loc: 
            num_target_samples = int(num_target_samples*0.9)

        samples = self.env_sampler.sample((num_target_samples,))

        if self.optimize_samples: 
            # Start optimization
            samples.requires_grad_(True)
            optimizer = torch.optim.Adam([samples],lr=0.05) 
            for _ in range(10): 
                loss = self.kernel_loss(samples)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            samples.requires_grad_(False)
        
        if self.sample_near_current_loc: 
            nearby_samples = self.loc_sampler.sample((int(num_target_samples/0.9*0.1),)) + self.robot.state[self.explr_locs].clone()
            samples = torch.vstack([samples,nearby_samples])

        if self.add_recent_history:
            samples = torch.vstack([samples,recent[:,self.explr_locs]])

        if self.test_corners:
            samples = torch.vstack([samples, self.corner_samples])

        # Sample Trajectory History
        traj_history = self.memory_buffer.sample(num_traj_samples)
        
        ## pad history?
        # traj_history_pad = torch.ones((num_traj_samples,self.robot.num_states),dtype=self.dtype)*self.barr_lim[:,1]*100.
        # traj_history_pad[:traj_history.shape[0]] = traj_history

        nu = torch.ones(1) 
        # nu = torch.as_tensor(len(traj_history) + 1)
        return samples, traj_history, nu

    @torch.no_grad()
    def forward(self,idx):
        # Make sure planner state is the same as robot state
        x = self.planner.reset(self.robot.state.clone())
        u_tmp = self.policy.reset(x,self.u.clone(),idx)

        grad_list = []
        traj_list = []

        # Forward Pass
        for t in range(self.horizon):
            # Get default control
            u_tmp[t] = self.policy(x)
            # Calculate derivatives
            A, B = self.planner.get_lin(x, u_tmp[t])
            dmudx = self.policy.dx(x, u_tmp[t])
            dbarrdx = self.barrier.dbarr(x) # *0.5**t # receding
            # Store for backward iteration
            grad_list.append((A,B,dbarrdx,dmudx))
            traj_list.append(x)
            # Step state forward
            x = self.planner.step(u_tmp[t])
        return u_tmp, grad_list, torch.vstack(traj_list)

    @torch.no_grad()
    def backward(self,samples,p,q,nu,grad_list,traj_list):
        rho = torch.zeros_like(self.planner.state)
        importance_ratio = p/q #(q+1e-1)
        du_list = torch.zeros_like(self.u)
        djdlam = torch.zeros(self.horizon,dtype=self.dtype)

        for t in reversed(range(self.horizon)):
            A,B,dbarrdx,dmudx = grad_list[t]
            x = traj_list[t]
            dgdx = self.kldiv_grad_vec(x, samples, self.explr_locs, self.std, importance_ratio,nu) #*nu/self.horizon
            rho = rk4_integrate(self.rho_dot,-self.planner.dt,rho,[dgdx,*grad_list[t]])
            du = -self.R_inv@B.T@rho
            du_list[t] = du
            if self.ctrlAppSearch:
                djdlam[t] = rho@B@du # really rho.T @ B @ du but shape of rho is (4,) so not needed / would have to unsqueeze to transpose

        return du_list, djdlam
    
    def get_target_dist(self,samples,temp,uniform=False,plot=False):
        # Get Target Distribution
        outside_bounds = ((samples < self.robot_lim[:,0]) | (samples > self.robot_lim[:,1])).sum(1).gt(0)
        if uniform:
            p = self.target_dist.init_uniform_grid(samples.clone().to(self.target_dist.device)).squeeze().to(device=self.device)
            # p[outside_bounds] = 0.
            p = renormalize(p)
        elif self.use_prior:
            p = self.prior_dist.pdf_torch(samples.clone().to(self.prior_dist.device)).squeeze().to(device=self.device)
            p = renormalize(p)
        else:
            p = self.target_dist.pdf_torch(samples.clone().to(self.target_dist.device)).squeeze().to(device=self.device)
        # p = renormalize(p)

        if self.average_prior: 
            p_prior = self.prior_dist.pdf_torch(samples.clone().to(self.prior_dists.device)).squeeze().to(device=self.device)
            p += renormalize(p_prior)

        if self.weight_env or self.weight_temp or plot: 
            if len(self.memory_buffer) > 0:
                traj = self.memory_buffer.get_all()
                spread = traj_spread_vec(traj, samples, self.explr_idx, self.std, nu=1.)
                spread /= torch.max(spread)
                spread[outside_bounds] = 1.
            else: 
                spread = torch.zeros(1)
            if self.weight_env and not plot: 
                p += (1-spread)*p.min() 
            elif self.weight_temp or plot: 
                spread_temp = torch.mean(spread)
                p = p**spread_temp            
            p = renormalize(p)

        p = p**temp # temperature param
        return p


    def kldiv_planner(self, num_target_samples, num_traj_samples, temp=1.0):

        samples, traj_history, nu = self.get_samples(num_target_samples,num_traj_samples)

        with torch.no_grad():

            p = self.get_target_dist(samples,temp,uniform=self.uniform_tdist)
            q_base = self.traj_footprint_vec_jit(traj_history.clone(),samples.clone(),self.explr_locs,self.std.clone(),nu)
            if len(traj_history) == 0:
                q_base = torch.zeros_like(q_base)

            last_cost = self.get_cost(samples.clone(), p.clone(), q_base.clone(), traj_history.clone(), self.u.clone(), self.u.clone())
            last_tapp = []
            traj_samples = traj_history.clone()
            q = renormalize(q_base.clone())

            for idx in range(self.num_iters_per_step):
                # Forward Pass
                u_tmp, grad_list, traj_list = self.forward(idx)

                # Get Trajectory Distribution
                last_traj_samples = traj_samples.clone()
                last_q = q.clone()

                traj_samples = torch.vstack([traj_history,traj_list]).to(self.dtype)
                q_iter = self.traj_footprint_vec_jit(traj_list.clone(),samples.clone(),self.explr_locs,self.std.clone(),nu)
                q = renormalize(q_base + q_iter)

                # Backward Pass
                du, djdlam = self.backward(samples.clone(),p.clone(),q.clone(),nu,grad_list,traj_list)

                if self.saturate:
                    u_star = self.saturate_control(u_tmp + self.alpha*du)
                else:
                    u_star = torch.clamp(u_tmp + self.alpha*du,*self.control_lim.T)

                # find application time and duration
                if self.ctrlAppSearch:
                    if self.full_cost:
                        u_tmp_full = self.u.expand(self.horizon,*self.u.shape).clone()
                        u_tmp_full[torch.arange(self.horizon),torch.arange(self.horizon),:] = u_star.clone()
                        if not self.rot_states:
                            cost = self.batched_cost(samples.expand(self.horizon,*samples.shape).clone(),
                                                    p.expand(self.horizon,*p.shape).clone(),
                                                    q_base.expand(self.horizon,*q_base.shape).clone(),
                                                    traj_history.expand(self.horizon,*traj_history.shape).clone(),
                                                    u_tmp_full,
                                                    self.u.expand(self.horizon,*self.u.shape).clone()
                                                    )
                        else:
                            cost = torch.zeros(self.horizon)
                            for t_mod,ut_tmp in enumerate(u_tmp):
                                # only update one control
                                # forward simulate & get cost
                                cost[t_mod] = self.get_cost(samples.clone(), p.clone(), q_base.clone(), traj_history.clone(),ut_tmp, self.u.clone())
                        cost = renormalize(cost)-1
                        djdlam = cost
                    # djdlam = djdlam * (0.9**torch.arange(self.horizon)) # receding
                    # djdlam[:idx]=1e6
                    t_app = torch.argmin(djdlam).item()
                    if (djdlam[t_app] < 0): #  and not(t_app == last_tapp): # only update if cost is negative
                        last_tapp = t_app
                        # print(djdlam[t_app])
                        # just do one time step for now (fixed lambda)
                        u_app = u_star[t_app]
                        if self.fixed_lam:
                            u_tmp[t_app:t_app+self.lam] = u_app.clone()
                        else:
                            tau, success = self.line_search(t_app,u_app.clone(),p.clone(),q_base.clone(),samples.clone(),traj_history.clone(),idx=idx,J0=last_cost)
                            if success: 
                                u_tmp[tau[0]:tau[1]] = u_app.clone()
                    else: # don't do any more loops if not changing anything
                        # print('got djdlam break')
                        q = last_q.clone()
                        traj_samples = last_traj_samples.clone()
                        break
                else:
                    u_tmp = u_star

                ## check that cost actually decreased before saving
                cost = self.get_cost(samples.clone(), p.clone(), q_base.clone(), traj_history.clone(), u_tmp.clone(), self.u.clone())
                if (idx > 0) and (last_cost <= cost):
                    # print('got cost break',idx,(last_cost-cost))
                    q = last_q.clone()
                    traj_samples = last_traj_samples.clone()
                    break
                last_cost = cost.clone()
                self.u = u_tmp
            # check for nans
            self.u = torch.nan_to_num(self.u)

            # forward simulate
            x = self.planner.reset(self.robot.state.clone())
            self.last_plan = torch.vstack([x]+[self.planner.step(ut) for ut in self.u])

            if self.plot_data is not None:
                if use_fut and (self.plot_extra or self.plot_smooth):
                    self.update_plot_fut = torch.jit._fork(self.update_plots,traj_samples,samples,p,q,temp,nu)
                else: 
                    self.update_plots(traj_samples,samples,p,q,temp,nu)

    @torch.no_grad()
    def rho_dot(self,rho,grads):
        dgdx,dfdx,dfdu,dbarrdx,dmudx = grads # dgdx, A, B, dbarrdx, dmudx
        return dgdx - dbarrdx - (dfdx + dfdu @ dmudx).T @ rho

    @torch.no_grad()
    def check_plots(self):
        if use_fut and (self.plot_extra or self.plot_smooth):
            # start = time.time()
            torch.jit._wait(self.update_plot_fut)
            # print(start-time.time())

    @torch.no_grad()
    def update_plots(self,traj_samples,samples,p,q,temp,nu):
        # generate extra plots
        tmp_pplot_samples = self.robot.state[self.explr_locs].expand_as(samples).clone()
        if self.plot_extra:
            self.extra_pplot = []
            self.extra_qplot = []
            for pi in self.all_plot_idx:
                # Get Target Distribution to plot
                pplot_samples = tmp_pplot_samples.clone()
                pplot_samples[:,pi] = samples[:,pi].clone()
                pplot = self.get_target_dist(pplot_samples,temp,plot=True)

                # Get Traj Distribution to plot
                qplot = self.traj_footprint_vec_jit(traj_samples.clone(),pplot_samples,self.explr_locs,self.std_plot,nu)
                qplot = renormalize(qplot)

                if self.test_corners:
                    self.extra_pplot.append(pplot)
                    self.extra_qplot.append(qplot)
                else:
                    self.extra_pplot.append(torch.hstack([pplot,self.corners*torch.min(pplot)]))
                    self.extra_qplot.append(torch.hstack([qplot,self.corners*torch.min(qplot)]))
        elif self.plot_smooth: 
            if single:
                # Get Target Distribution to plot
                pplot_samples = tmp_pplot_samples
                pplot_samples[:,self.plot_idx] = samples[:,self.plot_idx].clone()
                pplot = self.get_target_dist(pplot_samples,temp,plot=True)
                
                # Get Traj Distribution to plot
                qplot = self.traj_footprint_vec_jit(traj_samples.clone(),pplot_samples,self.explr_locs,self.std_plot,nu)
                qplot = renormalize(qplot)
            else:
                pplot_samples = []
                for offset in [-0.2,-0.1,-0.05,0.,0.05,0.1,0.2]:
                    pplot_samples_tmp = tmp_pplot_samples.clone() + offset
                    pplot_samples_tmp[:,self.plot_idx] = samples[:,self.plot_idx].clone()
                    pplot_samples.append(pplot_samples_tmp)
                pplot_samples = torch.vstack(pplot_samples)

                # Get Target Distribution to plot
                pplot = self.get_target_dist(pplot_samples,temp,plot=True).reshape(7,-1).mean(0)
                
                # Get Traj Distribution to plot
                qplot = self.traj_footprint_vec_jit(traj_samples.clone(),pplot_samples,self.explr_locs,self.std_plot,nu)
                qplot = renormalize(qplot.reshape(7,-1).mean(0))

            if self.test_corners:
                self.extra_pplot = pplot
                self.extra_qplot = qplot
            else: 
                self.extra_pplot = torch.hstack([pplot,self.corners*torch.min(pplot)])
                self.extra_qplot = torch.hstack([qplot,self.corners*torch.min(qplot)])
        if self.uniform_tdist:
            p = self.get_target_dist(samples,temp,uniform=False,plot=True)

        # save for plotting
        if self.test_corners:
            self.plot_data[0] = samples.clone()
            self.plot_data[1] = p.clone()
            self.plot_data[2] = q.clone()
        else:
            # add corners for plotting only
            self.plot_data[0] = torch.vstack([samples, self.corner_samples])
            self.plot_data[1] = torch.hstack([p,self.corners*torch.min(p)])
            self.plot_data[2] = torch.hstack([q,self.corners*torch.min(q)])
        self.plot_data[3] = self.last_plan[:,self.explr_locs].clone()
        if self.plot_extra:
            self.plot_data[4] = self.extra_pplot[self.desired_plot_idx].clone()
            self.plot_data[5] = self.extra_qplot[self.desired_plot_idx].clone()
        elif self.plot_smooth: 
            self.plot_data[4] = self.extra_pplot.clone()
            self.plot_data[5] = self.extra_qplot.clone()
        else: 
            self.plot_data[4] = self.plot_data[1].clone()
            self.plot_data[5] = self.plot_data[2].clone()
        p_tmp = cost_norm(p)
        q_tmp = cost_norm(q)
        D_KL = torch.sum(p_tmp * torch.log(p_tmp/q_tmp)) 
        self.plot_data[6] = D_KL

        # print(self.plot_data[1].min(),self.plot_data[1].max(),self.plot_data[4].min(),self.plot_data[4].max())

    @torch.no_grad()
    def get_cost(self, samples, p, q_base, traj_history, u_test, u_def, receding_barrier=False):
        # forward simulate
        x = self.planner.reset(self.robot.state.clone())
        traj_list = torch.vstack([self.planner.step(ut) for ut in u_test])
        # Get Trajectory Distribution
        N = torch.ones(1) # torch.as_tensor(1+len(traj_history))
        q_iter = self.traj_footprint_vec(traj_list,samples,self.explr_locs,self.std,N)
        q = renormalize(q_base + q_iter)
        # get cost
        # D_KL = - torch.sum(p * torch.log(q)) / N
        p = cost_norm(p)
        q = cost_norm(q)
        D_KL = torch.sum(p * torch.log(p/q)) / N
        ## add barrier and ctrl if desired 
        # udiff = u_test - u_def
        # control_cost = torch.tensordot(udiff@self.R_inv, udiff) # torch.sum(u[t] @ R @ u[t].T)
        control_cost = 0.
        if receding_barrier: 
            gamma = 0.5**torch.arange(self.horizon)
        else: 
            gamma = 1. 
        barrier_cost = torch.sum(self.barrier(traj_list)*gamma)
        cost = D_KL + control_cost + barrier_cost # *len(p)
        return cost

    @torch.no_grad()
    def line_search(self,t_app,u_app,p,q_base,samples,traj_history,idx=0,J0=None,max_app_dur=5):
        if t_app == 0 or t_app == self.horizon-1:
            lam = np.min([self.horizon,max_app_dur])
        elif t_app == idx: 
            lam = np.min([self.horizon-t_app,max_app_dur])
        else:
            lam = np.min([t_app-idx,self.horizon-t_app-idx,int(np.ceil(max_app_dur/2))])
        lam = np.max([lam,1])
        if J0 is None:
            J0 = self.get_cost(samples.clone(), p.clone(), q_base.clone(), traj_history.clone(), self.u.clone(), self.u.clone())
        Jn = J0*2
        tau_i, tau_f = [idx,lam]
        done = False
        while not(done) and (lam > 0):
            tau_last = [tau_i,tau_f]
            Jn_last = Jn
            # get window for this test
            if t_app == idx:
                tau_i = t_app
                tau_f = lam+1
            elif t_app == self.horizon-1: 
                tau_i = lam-1
                tau_f = t_app
            else:
                tau_i = t_app - lam
                tau_f = t_app + lam+1

            # Forward Pass
            tmp_u = self.u.clone()
            tmp_u[tau_i:tau_f] = u_app
            Jn = self.get_cost(samples.clone(), p.clone(), q_base.clone(), traj_history.clone(), tmp_u, self.u.clone())

            lam -= 1
            if (Jn_last < J0) and (Jn > Jn_last):
                done = True
        if (not done) and (Jn < J0): 
            tau_last = [tau_i,tau_f]
            done = True 
        return tau_last, done


if __name__ == "__main__":
    """
    note: if you're getting import erros like > ImportError: attempted relative import with no known parent package <
    you should try running this script as a module e.g $ python -m scripts.control_torch.klerg
    """
    
    import matplotlib.pyplot as plt

    class uniform_dist(object):
        def __init__(self):
            self.device='cpu'
            pass

        def pdf(self, x):
            x = torch.as_tensor(x) 
            pdf = self.pdf_torch(x)
            return pdf.numpy()

        def pdf_torch(self, x):
            # assert len(x.shape) > 1, 'Input needs to be a of size N x n'
            # val = torch.ones(x.shape[0])
            val = x.sum(1)**0
            val = val / torch.sum(val)
            val += 1e-5
            return val
        
    # target = uniform_dist()

    states = 'xyXY'
    plot_states = 'xy'
    target = dummyTestDist(torch.FloatTensor([-0.8,0.,0.9,0.]), torch.diag(torch.FloatTensor([0.06,1.,0.5,1.]))) 

    # states = 'yY'
    # plot_states = 'yY'
    # target = dummyTestDist(torch.FloatTensor([0.,1.4]), torch.diag(torch.FloatTensor([0.1,0.1]))) 
    
    vel_states=not(states.lower() == states)
    num_states = len(states)
    if vel_states: 
        num_ctrls = int(num_states/2)
    else: 
        num_ctrls = num_states
    x0 = np.array([*np.random.uniform(-1,1,size=(num_ctrls)), *np.zeros(num_ctrls)])
    target_samps = 3000
    horizon = 10
    
    ctrl_lim = [-1.5,1.5]
    figs = []
    robot = Robot(x0=x0, robot_lim=np.array([[-1.,1.]]*num_ctrls+[ctrl_lim]*num_ctrls*vel_states), 
                  robot_ctrl_lim=np.array([ctrl_lim]*num_ctrls), explr_idx=np.arange(num_states),horizon=horizon,
                  states=states, buffer_capacity=10000, R=0.05, target_dist=target,plot_data=True,plot_extra=True,
                  plot_states=plot_states,vel_states=vel_states,explr_robot_lim_scale=1.15)
    robot.test()
    path = []
    num_steps = 300
    for i in range(num_steps):
        state,vel,cmd = robot.step(num_target_samples= target_samps, num_traj_samples=num_steps,save_update=True)
        path.append(state)
    path = np.array(path)
    print(num_states,path.shape)

    samples = robot.plot_data[0]
    for fig_idx,ps in enumerate(robot.all_plot_states):
        q_plot = robot.extra_qplot[fig_idx]
        p_plot = robot.extra_pplot[fig_idx]
        plot_idx = robot.all_plot_idx[fig_idx]

        fig,ax = plt.subplots(2,2,figsize=(8,8),sharex=True,sharey=True)
        ax = ax.flatten()
        corner_samples = robot.all_corner_samples[fig_idx]
        samples[-corner_samples.shape[0]:] = corner_samples
        fig.suptitle(f'{target_samps} samples | horizon {horizon} | plot states {ps}')
        plot_data = [robot.plot_data[1],robot.plot_data[2],p_plot,q_plot]
        _min,_max = torch.amin(torch.stack(plot_data)),torch.amax(torch.stack(plot_data))
        for axs,title,data in zip(ax, ['target dist','trajectory dist','other targ dist','other tdist'],plot_data):
            axs.tricontourf(*samples[:,plot_idx].T, data, levels=30, vmin=_min, vmax=_max)
            axs.set_title(title)
            axs.plot(path[0,plot_idx[0]], path[0,plot_idx[1]], 'r*')
            axs.plot(path[:,plot_idx[0]], path[:,plot_idx[1]], 'k.')
            axs.plot(path[-1,plot_idx[0]], path[-1,plot_idx[1]], 'gs')
            axs.set_aspect('equal', 'box')
            axs.set_xlabel(ps[0])
        ax[0].set_ylabel(ps[1])
        # fig.tight_layout()
        figs.append(fig)

    plt.show(block=False)
    input('press any key to close all figures ')
    for f in figs: 
        plt.close(f)