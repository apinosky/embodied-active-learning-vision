#!/usr/bin/env python

import numpy as np
import numpy.random as npr
import itertools
from threading import Thread
import time 

from .memory_buffer import MemoryBuffer_numpy, AvoidDist
from .klerg_utils import renormalize, cost_norm,traj_spread_vec

from franka.franka_utils import ws_conversion
from .dynamics import SingleIntegratorEnv as dynamics

use_numba = True

if use_numba:
    from .klerg_utils import traj_footprint_vec_numba as traj_footprint_vec
else:
    from .klerg_utils import traj_footprint_vec

class DummyRobot(object):
    """ Dummy robot class that allows random controllers to plot target distribution and time averaged distribution """
    def __init__(self, x0, robot_lim, explr_idx, explr_robot_lim_scale = 1.0, target_dist=None, dt=0.1,
                        R=None, use_vel = True, pybullet=False,
                        horizon=10, buffer_capacity=100, std=0.05, std_plot=0.05, plot_data=False, plot_extra=False,
                        states='xy',plot_states='xy',tray_lim=None,robot_ctrl_lim=None, 
                        uniform_tdist=True, vel_states=False,use_magnitude=True):

        self.test_corners = False
        self.use_vel = not(uniform_tdist)
        self.use_prior = False
        self.pybullet = pybullet

        # Set up Workspace
        if not(len(robot_lim) == len(states)):
            robot_lim = [robot_lim]*len(states)
        self.explr_idx = explr_idx # these are the indexes based on inputs states (only used for plotting)
        self.ctrl_lims = np.array(robot_ctrl_lim)
        self.dt = dt
        self.states = states
        self.tray_lim = tray_lim
        self.robot_lim = np.array(robot_lim)
        self.vel_states = vel_states
        self.horizon = horizon+1
        self.plot_extra = plot_extra
        self.plot_smooth = True
        self.use_magnitude = use_magnitude

        if len(states) == len(plot_states): 
           self.plot_extra = False 
           self.plot_smooth = False 

        # Set up Target Distribution
        self.target_dist=target_dist
        self.dtype=np.float32

        if self.vel_states: # vel already accounted for in double integrator
            tmp = [[idx,s]  for idx,s in enumerate(states) if s==s.lower()]
            non_vel_locs,tmp_states = map(np.stack,zip(*tmp))
            states = ''.join(tmp_states.tolist())
            self.non_vel_locs = non_vel_locs
            self.vel_locs = [idx for idx,s in enumerate(self.states) if s==s.upper()]
        else: 
            self.non_vel_locs = list(range(len(states)))
            self.use_magnitude = False

        # Set up robot
        self.robot = dynamics(dt=dt,x0=np.array(x0)[self.non_vel_locs],states=states)
        self.explr_locs = [idx for idx,s in enumerate(states.lower()+states.upper()) if s in self.states] # these are the indexes based on dynamics
        self.robot_lim_NoVel = self.robot_lim[self.non_vel_locs]
        self.last_vel = np.zeros(self.robot.num_actions)


        # set up plotting
        self.lims = np.array(robot_lim)
        self.lims += np.tile(np.array([[-1.,1.]]),(len(self.lims),1))*(self.lims[:,[1]]-self.lims[:,[0]])*(explr_robot_lim_scale-1.)/2.
        if self.use_magnitude:
            self.lims[self.vel_locs,0] = 0.

        self.std = np.array([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std
        # self.std = np.array([1. if state in 'xyz' else (2. if state in 'rpw' else 0.5) for state in self.states],dtype=self.dtype)*std
        self.std_plot = np.array([1. if state.lower() == state else 5. for state in self.states],dtype=self.dtype)*std_plot
        self.memory_buffer = MemoryBuffer_numpy(buffer_capacity,self.robot.num_states*(2**self.vel_states),dtype=self.dtype)

        self.plot_data = plot_data
        self.plot_states = plot_states

    def setup_plotting(self,num_samples): 
        if self.plot_data:
            # self.plot_data = [0]*6 # placeholder for [samples tdist edist planned_traj]
            state = self.robot.state.copy()
            num_samples += 4
            samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_samples, len(self.explr_idx))).astype(self.dtype)
            dummy_qp = renormalize(np.ones(num_samples))
            dummy_locs = np.tile(state[self.explr_locs][None,:],(self.horizon,1))
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

    def step(self, num_target_samples, num_traj_samples, save_update=False, temp=1.):
        old_state = self.robot.state.copy()
        if self.use_vel:
            got_valid = False
            buffer = 0.05
            count = 0
            while not got_valid and count < 10: # first try actions
                vel = npr.uniform(*self.ctrl_lims.T,self.robot.num_actions)
                vel = (self.last_vel + vel)/2 # smoothing
                state = self.robot.step(vel,save=False)
                got_valid = ((self.robot_lim_NoVel[:,0] + buffer <= state)*(state<=self.robot_lim_NoVel[:,1] - buffer)).prod()
                count += 1
            if not got_valid: # if stuck use state
                state = npr.uniform(*self.robot_lim.T,len(self.robot_lim))
                vel = np.clip((state - old_state)/self.dt,*self.ctrl_lims.T)
            out = np.hstack([state,vel])
            self.last_vel = vel
        else:
            state = npr.uniform(*self.robot_lim_NoVel.T,len(self.robot_lim_NoVel))
            vel = np.clip((state - old_state)/self.dt,*self.ctrl_lims.T)
            out = np.hstack([state,vel])

        if save_update:
            self.save_update(out,save=True)

        if self.plot_data is not None:
            if self.plot_extra or self.plot_smooth:
                self.update_plot_t = Thread(target=self.update_plots, args=(num_target_samples,num_traj_samples,out,temp))
                self.update_plot_t.start()
            else: 
                self.update_plots(num_target_samples=num_target_samples, num_traj_samples=num_traj_samples,state=out,temp=temp)

        new_state = out[self.explr_locs]

        if self.use_magnitude:
            new_state[self.vel_locs] = np.abs(new_state[self.vel_locs])

        return new_state, vel, None

    def check_plots(self):
        if self.plot_extra or self.plot_smooth:
            # start = time.time()
            self.update_plot_t.join()
            # print(start-time.time())

    def save_update(self,full_state,force=0.,save=True):
        self.robot.reset(full_state[self.non_vel_locs])
        explr_state = full_state[self.explr_idx]
        if save:
            self.memory_buffer.push(explr_state)

    def test(self,num_target_samples = 100,N=10):
        traj_samples = npr.uniform(-1,1,size=(N,self.robot.num_states*(2**self.vel_states)))
        samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_target_samples, len(self.explr_idx))).astype(self.dtype)
        traj_footprint_vec(traj_samples,samples,self.explr_locs,self.std,N,dtype=self.dtype)
        self.setup_plotting(num_target_samples)

    def get_target_dist(self,samples,temp):
        # Get Target Distribution
        p = self.target_dist.pdf(samples).squeeze()
        p = p**temp # temperature param
        # p = renormalize(p)

        outside_bounds = ((samples < self.robot_lim[:,0]) | (samples > self.robot_lim[:,1])).sum(1) > 0 
        if len(self.memory_buffer) > 0:
            traj = self.memory_buffer.get_all()
            spread = traj_spread_vec(traj, samples, self.explr_idx, self.std, nu=1.,dtype=self.dtype)
            spread /= np.max(spread)
            spread[outside_bounds] = 1.
        else: 
            spread = np.zeros(1)
        spread_temp = np.mean(spread)
        p = p**spread_temp            

        p = renormalize(p)
        return p

    def update_plots(self, num_target_samples, num_traj_samples,state,temp=1):

        # Sample Target Distribution
        samples = npr.uniform(self.lims[self.explr_idx,0], self.lims[self.explr_idx,1],size=(num_target_samples, len(self.explr_idx))).astype(self.dtype)
        if self.test_corners:
            samples = np.vstack([samples, self.corner_samples])

        p = self.get_target_dist(samples,temp)

        # Sample Trajectory Distribution
        traj_samples = self.memory_buffer.sample(num_traj_samples).astype(self.dtype)
        nu = 1 #  len(traj_samples) + 1
        q = traj_footprint_vec(traj_samples,samples,self.explr_locs,self.std,nu,dtype=self.dtype)
        if len(traj_samples) == 0:
            q = np.zeros_like(q)
        q = np.clip(q,1e-6,None)
        q = renormalize(q)

        # extra plots 
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
                qplot = np.clip(qplot,1e-6,None)
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

        # save for plotting
        if self.test_corners:
            self.plot_data[0] = samples.copy()
            self.plot_data[1] = p.copy()
            self.plot_data[2] = q.copy()
        else:
            self.plot_data[0] = np.vstack([samples, self.corner_samples])
            self.plot_data[1] = np.hstack([p,self.corners*np.min(p)])
            self.plot_data[2] = np.hstack([q,self.corners*np.min(q)])
        self.plot_data[3] = np.tile(state[self.explr_locs][None,:],(self.horizon,1)).astype(self.dtype)
        if self.plot_extra:
            self.plot_data[4] = self.extra_pplot[self.desired_plot_idx]
            self.plot_data[5] = self.extra_qplot[self.desired_plot_idx]
        elif self.plot_smooth:
            self.plot_data[4] = self.extra_pplot
            self.plot_data[5] = self.extra_qplot
        else: 
            self.plot_data[4] = self.plot_data[1]
            self.plot_data[5] = self.plot_data[2]
        
        p_tmp = cost_norm(p)
        q_tmp = cost_norm(q)
        D_KL = np.sum(p_tmp * np.log(p_tmp/q_tmp)) 
        self.plot_data[6] = D_KL

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
    robot = DummyRobot(x0=x0, robot_lim=[[-1.,1.]]*num_states, robot_ctrl_lim=[[-1.,1.]]*num_states, explr_idx=np.arange(num_states), buffer_capacity=10000, R=0.01, target_dist=target, plot_data=True,states=states)
    robot.test()
    path = []
    num_steps = 300
    for i in range(num_steps):
        state,vel,cmd = robot.step(num_target_samples= 500, num_traj_samples=num_steps,save_update=True) 
        path.append(state)
    path = np.array(path)

    fig,ax = plt.subplots(1,2,figsize=(6,4),sharex=True,sharey=True)
    samples = robot.plot_data[0]
    fig.suptitle(f'plot states {robot.plot_states}')
    for axs,title,data in zip(ax,['target dist','trajectory dist'],[robot.plot_data[1],robot.plot_data[2]]):
        axs.tricontourf(*samples[:,robot.plot_idx].T, data, levels=30)
        axs.set_title(title)
        axs.plot(path[0,robot.plot_idx[0]], path[0,robot.plot_idx[1]], 'r*')
        axs.plot(path[:,robot.plot_idx[0]], path[:,robot.plot_idx[1]], 'k.')
        axs.set_aspect('equal', 'box')
    plt.show()

