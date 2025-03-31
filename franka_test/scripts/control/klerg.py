#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import itertools
from termcolor import cprint
import yaml
import time
from threading import Thread

from .memory_buffer import MemoryBuffer_numpy, AvoidDist
from .dynamics import rk4_integrate
from .barrier import *
from .default_policies import *
from .klerg_utils import *

import sys, os
base_path = os.path.dirname(os.path.abspath(__file__))
from franka.franka_utils import ws_conversion, find_non_vel_locs

use_numba = True

if use_numba:
    from .klerg_utils import kldiv_grad_vec_numba as kldiv_grad_vec
    from .klerg_utils import traj_footprint_vec_numba as traj_footprint_vec
else:
    from .klerg_utils import kldiv_grad_vec, traj_footprint_vec

class Robot(object):
    """ Robot class that runs the KL-Erg MPC Planner """
    def __init__(self, x0, robot_lim, explr_idx, explr_robot_lim_scale = 1.0, target_dist=None, dt=0.1,
                    R=0.01, use_vel = True, pybullet=False,
                    horizon=10, buffer_capacity=100, std=0.05, std_plot=0.05, plot_data=False, plot_extra=False,
                    states='xy',plot_states='xy',tray_lim=None,robot_ctrl_lim=None,
                    uniform_tdist=False, vel_states=False, use_magnitude=False):

        self.load_yaml() 

        self.target_dist=target_dist
        self.dtype=np.float32
        self.use_prior = False
        self.pybullet = pybullet

        # Set up Workspace
        self.robot_lim = robot_lim
        self.explr_idx = explr_idx # these are the indexes based on inputs states
        self.states = states
        self.robot_ctrl_lim = robot_ctrl_lim
        self.uniform_tdist = uniform_tdist
        self.vel_states = vel_states
        self.horizon = horizon
        self.plot_extra = plot_extra
        self.plot_smooth = True
        self.use_magnitude = use_magnitude
        self.use_vel = use_vel
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
            # assert np.all(self.robot_lim[rpw] == self.tray_lim[rpw]),'robot_lim needs to use real angles for roll,pitch,yaw to use current DoubleIntegratorRollEnv implementation'
            if not(np.all(self.robot_lim[rpw] == self.tray_lim[rpw])):
                extra_args['rot_to_angles_fn'] = lambda x: ws_conversion(x,self.robot_lim[rpw],self.tray_lim[rpw])
                extra_args['angles_to_rot_fn'] = lambda x: ws_conversion(x,self.tray_lim[rpw],self.robot_lim[rpw])
            from .dynamics import DoubleIntegratorRollEnv as dynamics
        else:
            self.rot_states = False
            if self.use_magnitude:
                from .dynamics import DoubleIntegratorSpeedEnv as dynamics
                x0 = np.hstack([x0,np.zeros(len(self.non_vel_locs))])
            else:
                from .dynamics import DoubleIntegratorEnv as dynamics

        # Set up robot & planner
        dt_scale = 1. if self.use_vel else 4.
        self.robot = dynamics(dt=dt*dt_scale,x0=x0,states=states,**extra_args)
        self.explr_locs = np.array([idx for idx,s in enumerate(self.robot.states) if s in self.states]) # these are the indexes based on dynamics
        self.planner = dynamics(dt=dt,x0=x0,states=states,**extra_args)

        if 'b' in self.robot.states:
            self.bv_idx = self.robot.states.rfind('B')

        # Set up sampling 
        self.lims = np.array(self.robot_lim)
        self.lims += np.tile(np.array([[-1.,1.]]),(len(self.lims),1))*(self.lims[:,[1]]-self.lims[:,[0]])*(explr_robot_lim_scale-1.)/2.
        if self.use_magnitude:
            self.lims[self.vel_locs,0] = 0.

        # Set up other parameters
        self.num_iters_per_step = max(1,int(self.pct_of_horizon_for_inner_loop*self.horizon))
        self.std = np.array([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std
        # self.std = np.array([1. if state in 'xyz' else (2. if state in 'rpw' else 0.5) for state in self.states],dtype=self.dtype)*std
        self.std_plot = np.array([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std_plot

        if isinstance(R,(int,float)):
            R = [R]*self.robot.num_actions
        self.R_inv = np.linalg.inv(np.diag(R))
        self.u = np.zeros((self.horizon, self.planner.num_actions),dtype=self.dtype)
        self.memory_buffer = MemoryBuffer_numpy(buffer_capacity,self.planner.num_states,dtype=self.dtype)
        self.control_lim = np.array([[-1.0,1.0] for state in states],dtype=self.dtype)
        # self.control_lim = np.array([[-1.0,1.0] if state in 'xyzb' else [-0.5,0.5] for state in states],dtype=self.dtype)

        # set up default policy
        DefaultPolicy = eval(self.DefaultPolicy)
        self.policy = DefaultPolicy(self.planner,self.horizon)

        # other setup
        self.plot_data = plot_data
        self.plot_states = plot_states
        self.barrier,self.barr_lim = setup_barrier(states,self.robot_lim,self.robot_ctrl_lim,self.non_vel_locs,self.dtype,extra_args,self.rot_states)

        self.count = 0
        cprint('[klerg] setup complete','cyan')
        
    def load_yaml(self): 
        with open(base_path+'/robot_config.yaml') as f:
            yaml_config = yaml.load(f,Loader=yaml.FullLoader)
        for k, v in yaml_config.items():
            setattr(self, k, v)

    def setup_plotting(self,num_samples): 
        if self.plot_data:
            # self.plot_data = [0]*6 # placeholder for [samples tdist edist planned_traj]
            state = self.robot.state.copy()
            num_samples += 4
            samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_samples, len(self.explr_idx))).astype(self.dtype)
            dummy_qp = renormalize(np.ones(num_samples))
            dummy_locs = np.tile(state[self.explr_locs][None,:],(self.horizon+1,1))
            dummy_cost = np.array([1000.])
            self.plot_data = [samples] + [dummy_qp]*2 + [dummy_locs] + [dummy_qp]*2 + [dummy_cost]

            # get all combinations of exploration states
            self.all_plot_states = [x[0]+x[1] for x in itertools.combinations(self.states,2)]
            self.all_plot_idx = [np.array([self.states.rfind(s) for s in ps]) for ps in self.all_plot_states] # these are the indexes based on inputs states
            if any(np.hstack(self.all_plot_idx)==-1):
                raise ValueError('robot controller (klerg) did not find requested plot state')
            self.all_corner_samples = [self.get_corners(ps) for ps in self.all_plot_idx]
            
            # main plot states
            self.desired_plot_idx = np.argwhere(np.array(self.all_plot_states) == self.plot_states).item()
            self.plot_idx = self.all_plot_idx[self.desired_plot_idx]
            self.corner_samples = self.all_corner_samples[self.desired_plot_idx]
            self.corners = np.ones(len(self.corner_samples),dtype=self.dtype)
        else:
            self.plot_idx = np.array([self.states.rfind(s) for s in self.plot_states]) 
            self.plot_data = None
            self.test_corners = False
        self.last_plan = np.vstack([self.robot.state]+[self.robot.step(ut) for ut in self.u])

    def update_lims(self,idx,lims): 
        self.lims[idx] = lims
        if self.use_magnitude:
            self.lims[self.vel_locs,0] = 0.
        self.update_corners()
        if self.use_barrier:
            self.barrier.update_lims(self.lims.tolist()+self.robot_ctrl_lim.tolist())

    def update_corners(self):
        self.corner_samples = self.get_corners(self.plot_idx)

    def get_corners(self,plot_idx):
        corner_samples = np.array(list(itertools.product(*self.lims[plot_idx])),dtype=self.dtype)
        if len(self.explr_idx) > 2:
            corner_dim = len(self.explr_idx)
            tmp_corner_samples = np.zeros((corner_samples.shape[0],corner_dim),dtype=self.dtype)
            tmp_corner_samples[:,plot_idx] = corner_samples
            corner_samples = tmp_corner_samples
        return corner_samples

    def step(self, num_target_samples= 50, num_traj_samples=30,save_update=False, temp=1.0):
        self.kldiv_planner(num_target_samples= num_target_samples, num_traj_samples= num_traj_samples, temp=temp)

        ctrl = self.u[0].copy()
        if not save_update:
            state = self.robot.step(ctrl,save=False)
        else:
            state = self.robot.step(ctrl)
            self.save_update(state,save=True)
        vel = state[self.planner.num_actions:]
        return state[self.explr_locs], vel, ctrl

    def save_update(self,full_state,force=0.,save=True):
        if 'b' in self.states: 
            full_state[self.bv_idx] = self.last_plan[1][self.bv_idx] # not provided by ROS (always = 0)

        ''' find closest loc '''
        policy_idx = np.linalg.norm(self.last_plan - full_state,axis=1).argmin() # which is closest
        planned_state = self.last_plan[policy_idx]
        ''' update state '''
        vel_smoothing = 0.8
        full_state[self.planner.num_actions:] = vel_smoothing*full_state[self.planner.num_actions:] + (1-vel_smoothing)*planned_state[self.planner.num_actions:]
        x = self.robot.reset(full_state)

        ''' update control based on state '''
        self.u = self.policy.reset(x,self.u.copy(),-policy_idx)
        # self.last_plan = self.last_plan[policy_idx:]

        if save:
            self.memory_buffer.push(self.robot.state)


    def test(self,num_target_samples = 100,N=10):
        traj_samples = npr.uniform(-1,1,size=(N,self.planner.num_states))
        x = traj_samples[0]
        samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_target_samples, len(self.explr_idx))).astype(self.dtype)
        importance_ratio = traj_footprint_vec(traj_samples,samples,self.explr_locs,self.std,N,dtype=self.dtype)
        kldiv_grad_vec(x, samples, self.explr_locs, self.std, importance_ratio, 1.,dtype=self.dtype)
        self.setup_plotting(num_target_samples)

    def saturate_control(self,u,app_thresh=0.1):
        # r = np.arange(len(self.control_lim))
        # c = (u > 0).astype(int)
        # out = self.control_lim[r,c].copy()
        # out[np.abs(u) < app_thresh] = 0.
        out = np.tanh(u/app_thresh)*self.control_lim[:,1]
        return out

    def get_samples(self,num_target_samples,num_traj_samples):
        # Sample Target Distribution
        samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_target_samples, len(self.explr_idx))).astype(self.dtype)

        if self.add_recent_history:
            recent = self.memory_buffer.get_recent(self.horizon)
            samples[:len(recent)] = recent[:,self.explr_locs]
        if self.test_corners:
            samples = np.vstack([samples, self.corner_samples])

        # Sample Trajectory History
        traj_history = self.memory_buffer.sample(num_traj_samples)
        nu = 1 # len(traj_history) + 1
        return samples, traj_history, nu

    def forward(self,idx):
        # Make sure planner state is the same as robot state
        x = self.planner.reset(self.robot.state)
        u_tmp = self.policy.reset(x,self.u.copy(),idx)

        grad_list = []
        traj_list = []

        # Forward Pass
        for t in range(self.horizon):
            # Get default control
            u_tmp[t] = self.policy(x)
            # Calculate derivatives
            A, B = self.planner.get_lin(x, u_tmp[t])
            dmudx = self.policy.dx(x, u_tmp[t])
            dbarrdx = self.barrier.dbarr(x)
            # Store for backward iteration
            grad_list.append((A,B,dbarrdx,dmudx))
            traj_list.append(x)
            # Step state forward
            x = self.planner.step(u_tmp[t])
        return u_tmp, grad_list, np.stack(traj_list)

    def backward(self,samples,p,q,nu,grad_list,traj_list):
        # Backwards pass
        rho = np.zeros(self.planner.state.shape,dtype=self.dtype)
        importance_ratio = p/(q+1e-1)

        du_list = np.zeros_like(self.u)
        djdlam = np.zeros(self.horizon)

        for t in reversed(range(self.horizon)):
            A,B,dbarrdx,dmudx = grad_list[t]
            x = traj_list[t]
            dgdx = kldiv_grad_vec(x, samples, self.explr_locs, self.std, importance_ratio,nu,dtype=self.dtype) #*nu/self.horizon
            rho = rk4_integrate(self.rho_dot,-self.planner.dt,rho,[dgdx,*grad_list[t]])
            du = -self.R_inv@B.T@rho
            du_list[t] = du
            if self.ctrlAppSearch:
                djdlam[t] = rho.T@B@du
        return du_list, djdlam
    
    def get_target_dist(self,samples,temp,uniform=False):
        # Get Target Distribution
        if uniform:
            p = self.target_dist.init_uniform_grid(samples).squeeze()
            p = renormalize(p)
        else:
            p = self.target_dist.pdf(samples).squeeze()
        p = p**temp # temperature param
        # p = renormalize(p)
        p_count = 1.

        if self.weight_env: 
            if len(self.memory_buffer) > 0:
                traj = self.memory_buffer.get_all()
                spread = traj_spread_vec(traj, samples, self.explr_idx, self.std*2., nu=1.)
                spread /= np.max(spread)
            else: 
                spread = 0.
            p += (1-spread)*p.min() # 0.25
            p_count += 1.

        # p /= p_count 
        p = renormalize(p)
        return p

    def kldiv_planner(self, num_target_samples, num_traj_samples, temp=1.0):
        samples, traj_history, nu = self.get_samples(num_target_samples,num_traj_samples)

        p = self.get_target_dist(samples,temp,uniform=self.uniform_tdist)
        q_base = traj_footprint_vec(traj_history.copy(),samples.copy(),self.explr_locs.copy(),self.std.copy(),nu,dtype=self.dtype)
        if len(traj_history) == 0:
            q_base = np.zeros_like(q_base)

        last_cost = self.get_cost(samples.copy(), p.copy(), q_base.copy(), traj_history.copy(),self.u.copy(), self.u.copy())
        last_tapp = []
        traj_samples = traj_history.copy()
        q = renormalize(q_base.copy())


        for idx in range(self.num_iters_per_step):
            # Forward Pass
            u_tmp, grad_list, traj_list = self.forward(idx)

            # Get Trajectory Distribution
            last_traj_samples = traj_samples.copy()
            last_q = q.copy()

            traj_samples = np.vstack([traj_history,traj_list]).astype(self.dtype)
            q_iter = traj_footprint_vec(traj_list.copy(),samples.copy(),self.explr_locs.copy(),self.std.copy(),nu,dtype=self.dtype)
            q = renormalize(q_iter + q_base)

            # check for invalid values
            if any(np.isnan(p)) or any(np.isinf(p)) or any(np.isinf(q)) or any(np.isnan(q)):
                print(np.sum(np.isnan(p)),np.sum(np.isinf(p)),np.sum(np.isnan(q)),np.sum(np.isinf(q)))

            # Backward Pass
            du, djdlam = self.backward(samples,p,q,nu,grad_list,traj_list)

            if self.saturate:
                u_star = self.saturate_control(u_tmp + self.alpha*du)
            else:
                u_star = np.clip(u_tmp + self.alpha*du,*self.control_lim.T)


            # find application time and duration
            if self.ctrlAppSearch:
                if self.full_cost:
                    cost = np.zeros(self.horizon)
                    for t_mod,u_mod in enumerate(u_star.copy()):
                        # only update one control
                        ut_tmp = self.u.copy()
                        ut_tmp[t_mod] = u_mod
                        # forward simulate & get cost
                        cost[t_mod] = self.get_cost(samples.copy(), p.copy(),  q_base.copy(),traj_history.copy(), ut_tmp, self.u.copy())
                    cost = renormalize(cost)-1
                    djdlam = cost
                # djdlam = djdlam * (0.9**np.arange(self.horizon))
                djdlam[:idx]=1e6
                t_app = np.argmin(djdlam)
                if (djdlam[t_app] < 0) and not(t_app == last_tapp): # only update if cost is negative
                    last_tapp = t_app
                    # print(djdlam[t_app])
                    # just do one time step for now (fixed lambda)
                    u_app = u_star[t_app]
                    if self.fixed_lam:
                        u_tmp[t_app:t_app+self.lam] = u_app
                    else:
                        tau, success = self.line_search(t_app.copy(),u_app.copy(),p.copy(),q_base.copy(),samples.copy(),traj_history.copy(),idx=idx,J0=last_cost)
                        if success:
                            u_tmp[tau[0]:tau[1]] = u_app
                else: # don't do any more loops if not changing anything
                    q = last_q.copy()
                    traj_samples = last_traj_samples.copy()
                    break
            else:
                u_tmp = u_star

            ## check that cost actually decreased before saving
            cost = self.get_cost(samples.copy(), p.copy(), q_base.copy(), traj_history.copy(), self.u.copy(), self.u.copy())
            if (idx > 0) and (last_cost <= cost):
                # print('got cost break',idx,(last_cost-cost))
                q = last_q.copy()
                traj_samples = last_traj_samples.copy()
                break
            last_cost = cost
            self.u = u_tmp

        # forward simulate
        x = self.planner.reset(self.robot.state.copy())
        self.last_plan = np.vstack([x]+[self.planner.step(ut) for ut in self.u])

        if self.plot_data is not None:
            if self.plot_extra or self.plot_smooth:
                self.update_plot_t = Thread(target=self.update_plots, args=(traj_samples,samples,p,q,temp,nu))
                self.update_plot_t.start()
            else: 
                self.update_plots(traj_samples,samples,p,q,temp,nu)


    def rho_dot(self,rho,grads):
        dgdx,dfdx,dfdu,dbarrdx,dmudx = grads # dgdx, A, B, dbarrdx, dmudx
        return dgdx - dbarrdx - (dfdx + dfdu @ dmudx).T @ rho

    def check_plots(self):
        if self.plot_extra or self.plot_smooth:
            # start = time.time()
            self.update_plot_t.join()
            # print(start-time.time())

    def update_plots(self,traj_samples,samples,p,q,temp,nu):
        # generate extra plots
        tmp_pplot_samples = np.broadcast_to(self.robot.state[self.explr_locs].copy(),samples.shape)
        if self.plot_extra:
            self.extra_pplot = []
            self.extra_qplot = []
            for pi in self.all_plot_idx:
                # Get Target Distribution to plot
                pplot_samples = tmp_pplot_samples.copy()
                pplot_samples[:,pi] = samples[:,pi].copy()
                pplot = self.get_target_dist(pplot_samples,temp)

                # Get Traj Distribution to plot
                qplot = traj_footprint_vec(traj_samples,pplot_samples,self.explr_locs,self.std_plot,nu,dtype=self.dtype)
                qplot = renormalize(qplot)            
                if self.test_corners:
                    self.extra_pplot.append(pplot)
                    self.extra_qplot.append(qplot)
                else:
                    self.extra_pplot.append(np.hstack([pplot,self.corners*np.min(pplot)]))
                    self.extra_qplot.append(np.hstack([qplot,self.corners*np.min(qplot)]))
        elif self.plot_smooth: 
            # Get Target Distribution to plot
            pplot_samples = tmp_pplot_samples.copy()
            pplot_samples[:,self.plot_idx] = samples[:,self.plot_idx].copy()
            pplot = self.get_target_dist(pplot_samples,temp)
            
            # Get Traj Distribution to plot
            qplot = traj_footprint_vec(traj_samples,pplot_samples,self.explr_locs,self.std_plot,nu,dtype=self.dtype)
            qplot = renormalize(qplot)
            if self.test_corners:
                self.extra_pplot = pplot
                self.extra_qplot = qplot
            else: 
                self.extra_pplot = np.hstack([pplot,self.corners*np.min(pplot)])
                self.extra_qplot = np.hstack([qplot,self.corners*np.min(qplot)])
        if self.uniform_tdist:
            p = self.get_target_dist(samples,temp)

        # save for plotting
        if self.test_corners:
            self.plot_data[0] = samples.copy()
            self.plot_data[1] = p.copy()
            self.plot_data[2] = q.copy()
        else:
            # add corners for plotting only
            self.plot_data[0] = np.vstack([samples, self.corner_samples])
            self.plot_data[1] = np.hstack([p,self.corners*np.min(p)])
            self.plot_data[2] = np.hstack([q,self.corners*np.min(q)])
        self.plot_data[3] = self.last_plan[:,self.explr_locs].copy()
        if self.plot_extra:
            self.plot_data[4] = self.extra_pplot[self.desired_plot_idx].copy()
            self.plot_data[5] = self.extra_qplot[self.desired_plot_idx].copy()
        elif self.plot_smooth:
            self.plot_data[4] = self.extra_pplot.copy()
            self.plot_data[5] = self.extra_qplot.copy()
        else: 
            self.plot_data[4] = self.plot_data[1].copy()
            self.plot_data[5] = self.plot_data[2].copy()
        p_tmp = cost_norm(p)
        q_tmp = cost_norm(q)
        D_KL = np.sum(p_tmp * np.log(p_tmp/q_tmp)) 
        self.plot_data[6] = D_KL

    def get_cost(self, samples, p, q_base, traj_history, u_test, u_def, receding_barrier=False):
        # forward simulate
        x = self.planner.reset(self.robot.state.copy())
        traj_list = np.vstack([self.planner.step(ut) for ut in u_test])
        # Get Trajectory Distribution
        N = 1 # 1+len(traj_history)
        q_iter = traj_footprint_vec(traj_list,samples,self.explr_locs,self.std,N,dtype=self.dtype)
        q = renormalize(q_base + q_iter)
        # get cost
        p = cost_norm(p)
        q = cost_norm(q)
        # D_KL = - np.sum(p * np.log(q)) / N
        D_KL = np.sum(p * np.log(p/q)) / N
        ## add barrier and ctrl if desired 
        # udiff = u_test - u_def
        # control_cost = np.tensordot(udiff @ self.R_inv,udiff,axes=2) # np.sum(u[t] @ R @ u[t].T)
        control_cost = 0.
        if receding_barrier: 
            gamma = 0.5**np.arange(self.horizon)
        else: 
            gamma = 1. 
        barrier_cost = np.sum(self.barrier(traj_list)*gamma)
        cost = D_KL + control_cost + barrier_cost #*len(p)
        return cost

    def line_search(self,t_app,u_app,p,q_base,samples,traj_history,idx=0,J0=None,max_app_dur=5):
        if t_app == 0 or t_app == self.horizon-1:
            lam = np.min([self.horizon,max_app_dur])
        elif t_app == idx: 
            lam = np.min([self.horizon-t_app,max_app_dur])
        else:
            lam = np.min([t_app,self.horizon-t_app,int(np.ceil(max_app_dur/2))])
        lam = np.max([lam,1])
        if J0 is None:
            J0 = self.get_cost(samples.copy(), p.copy(), q_base.copy(), traj_history.copy(), self.u.copy(), self.u.copy())
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
            tmp_u = self.u.copy()
            tmp_u[tau_i:tau_f] = u_app
            Jn = self.get_cost(samples.copy(), p.copy(), q_base.copy(), traj_history.copy(), tmp_u, self.u.copy())

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
    you should try running this script as a module e.g $ python -m scripts.control.klerg
    """
    
    import matplotlib.pyplot as plt

    class uniform_dist(object):
        def __init__(self):
            pass

        def pdf(self, x):
            val = np.ones(x.shape[0])
            val /= np.sum(val)
            val += 1e-5
            return val
    target = uniform_dist()

    states = 'xyz'
    num_states = len(states)
    x0 = np.array([*npr.uniform(-1,1,size=(num_states)), *np.zeros(num_states)])
    robot = Robot(x0=x0, robot_lim=np.array([[-1.,1.]]*num_states), robot_ctrl_lim=np.array([[-1.,1.]]*num_states), explr_idx=np.arange(len(states)), plot_data=True, states=states, buffer_capacity=10000, R=0.01, target_dist=target)
    robot.test()
    path = []
    num_steps = 300
    for i in range(num_steps):
        state,vel,cmd = robot.step(num_target_samples= 100, num_traj_samples=num_steps,save_update=True) 
        path.append(state)
    path = np.array(path)
     
    for plot_data_locs in [[1,2],[4,5]]:
        fig,ax = plt.subplots(1,2,figsize=(6,3),sharex=True,sharey=True)
        samples = robot.plot_data[0]
        fig.suptitle(f'plot states {robot.plot_states}')
        plot_data = [robot.plot_data[pd] for pd in plot_data_locs]
        _min,_max = np.amin(np.stack(plot_data)),np.amax(np.stack(plot_data))
        plot_idx = robot.plot_idx
        for axs,title,data in zip(ax,['target dist','trajectory dist'],plot_data):
            axs.tricontourf(*samples[:,plot_idx].T, data, levels=30, vmin=_min, vmax=_max)
            axs.set_title(title)
            axs.plot(path[0,plot_idx[0]], path[0,plot_idx[1]], 'r*')
            axs.plot(path[:,plot_idx[0]], path[:,plot_idx[1]], 'k.')
            axs.set_aspect('equal', 'box')
    plt.show()
