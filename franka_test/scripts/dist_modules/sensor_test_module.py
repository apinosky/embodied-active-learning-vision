#!/usr/bin/env python

save_checkpoint_models = False

########## global imports ##########
import torch
import numpy as np
import yaml
from scipy.stats import multivariate_normal as mvn

import time
from termcolor import cprint
import os

import rospy
from std_msgs.msg import Empty
from franka_test.srv import *

########## local imports ##########
from franka.franka_utils import ws_conversion
from .sensor_utils import SensorMainRosBase
from .utils import set_seeds

class SensorTest(SensorMainRosBase):
    def __init__(self,target_dist,num_steps,num_target_samples=1000,init_vel=True,explr_states=None,manual=False,explr_robot_lim_scale = 1.15):
        # modes
        self.explr_robot_lim_scale = explr_robot_lim_scale

        # params
        self.test_path = rospy.get_param('test_path', 'data/intensity/entklerg_0000/')
        self.pybullet = rospy.get_param('pybullet', False)
        print('*** Pybullet = ',self.pybullet,'***')

        # path to saved vars
        base_path = rospy.get_param('base_path', './')
        dir_path = base_path + '/' + self.test_path + '/'

        # load variables
        config_loc = dir_path + "/config.yaml"
        if os.path.exists(config_loc):
            with open(config_loc,"r") as f:
                params = yaml.load(f, Loader=yaml.FullLoader)
            self.map_dict(params)
            # initalize buffers
            self.tray_lim = np.array(self.tray_lim)
            self.robot_lim = np.array(self.robot_lim)
            self.tray_ctrl_lim = np.array(self.tray_ctrl_lim)
            self.robot_ctrl_lim = np.array(self.robot_ctrl_lim)
            self.dtype = eval(self.dtype)
        else: 
            from load_config import get_config
            args = get_config(print_output=False)
            self.map_dict(vars(args))

        self.dir_path = dir_path

        # seeds
        set_seeds(self.seed)
        self.explr_update = None

        # specify target dists
        self.traj_buffer_capacity = num_steps
        self.num_traj_samples = num_steps # make sure all steps are used for ergodic calculation
        self.num_target_samples = num_target_samples
        self.target_dist = target_dist

        # ros
        super(SensorTest, self).__init__(self.tray_lim, self.robot_lim, self.dir_path,self.states,self.plot_states)
        self.states_orig = self.states
        self.xinit_orig = self.xinit.copy()
        if init_vel:
            self.start_explr(explr_states=explr_states)
        else:
            self.use_pose()
        if not self.pybullet:
            time.sleep(0.5)
        rospy.loginfo_once("ready to run")
        self.manual = manual
        if self.manual:
            rospy.loginfo_once("you can now press e-stop and manually control robot arm")

    def start_robot(self,explr_states,robot_state=None):
        explr_idx = []
        for state in explr_states:
            explr_idx.append(self.states_orig.rfind(state))

        assert not((np.array(explr_idx)==-1).any()),'requested exploration state not present in states list'

        if robot_state is not None:
            self.xinit = robot_state
        else:
            self.xinit = np.array(self.xinit)[explr_idx]

        # Initialize KL-Erg Robot
        from control_torch.klerg import Robot
        x0 = np.hstack([*self.xinit, *np.zeros(len(self.xinit))])
        self.explr_idx = explr_idx
        non_vel_idx = self.update_states(explr_states)
        self.tray_lim = self.tray_lim[explr_idx]
        self.tray_ctrl_lim=self.tray_ctrl_lim[non_vel_idx]
        self.robot_lim = self.robot_lim[explr_idx]
        self.robot_ctrl_lim=self.robot_ctrl_lim[non_vel_idx]

        args = {}
        shared_vars = ['vel_states','states','dt','explr_idx','target_dist','horizon','tray_lim','robot_ctrl_lim','R','plot_states','explr_robot_lim_scale','use_magnitude','use_vel','std','pybullet']
        for key in shared_vars:
            args[key] = getattr(self,key)

        self.robot = Robot(x0=x0, robot_lim=self.robot_lim.copy(), 
                            buffer_capacity=self.traj_buffer_capacity, plot_data=True,
                            uniform_tdist=('unif' in self.explr_method), **args)

    def start_explr(self,robot_state=None,explr_states=None):
        start_vel_controller = rospy.Publisher("/switch_to_vel_controller",Empty,queue_size=1,latch=True)
        rospy.logwarn("switching controller")
        if self.pybullet: 
            self.send_cmd = lambda a,b: self.franka_env.velCallback(UpdateVelRequest(a,b))
        else:
            start_vel_controller.publish()
            rospy.wait_for_service('klerg_cmd')
            self.send_cmd = rospy.ServiceProxy('/klerg_cmd', UpdateVel)

        self.use_vel = True
        # self.rate = rospy.Rate(5)
        self.start_robot(explr_states,robot_state)
        self.reset_pub.publish()

    def use_pose(self):
        start_pose_controller = rospy.Publisher("/switch_to_pose_controller",Empty,queue_size=1,latch=True)
        rospy.logwarn("switching controller")
        if self.pybullet: 
            self.send_cmd = lambda a,b: self.franka_env.poseCallback(UpdateStateRequest(a,b))
        else:
            start_pose_controller.publish()
            rospy.wait_for_service('klerg_pose')
            self.send_cmd = rospy.ServiceProxy('/klerg_pose', UpdateState)
            time.sleep(1)

        self.use_vel = False
        # self.rate = rospy.Rate(1)
        self.reset_pub.publish()

    def _get_latest_data(self):
        ### get latest data from subscriber
        data,pos,full_pos,force,data_success = self.get_latest_msg()

        # Convert env state to klerg state
        robot_state = ws_conversion(pos, self.tray_lim, self.robot_lim)
        full_state = ws_conversion(full_pos, self.tray_full_lim, self.robot_full_lim) # double integrator format

        success = self.check_cmd(robot_state) and data_success

        return data,robot_state,full_state,force,success


    @torch.no_grad()
    def step(self,iter_step=0,pos=None):
        if self.use_vel:
            ### Plan in KL-Erg ###
            state, vel, action = self.robot.step(num_target_samples=self.num_target_samples, num_traj_samples=self.num_traj_samples, save_update=False)
            ### Step in Franka environment ###
            # Generate target position in env
            vel = ws_conversion(vel, self.robot_ctrl_lim, self.tray_ctrl_lim)
            vel = np.clip(vel,*np.array(self.tray_ctrl_lim).T)
            robot_vel = ws_conversion(vel, self.tray_ctrl_lim, self.robot_ctrl_lim)
            cmd = self.format_Twist_msg(vel)
        else:
            # Generate target position in env
            cmd = self.format_Pose_msg(pos)
            robot_vel = np.zeros(len(self.states))

        if self.brightness_idx >= 0: 
            brightness = state[self.brightness_idx]
        else: 
            brightness = -1.

        if not self.manual:
            try:
                pos_msg = self.send_cmd(cmd,brightness)
            except rospy.ServiceException as e:
                rospy.logwarn(f'pausing -- resolve the following ServiceException before resuming\nrospy.service.ServiceException:{e}')
                self.pause_pub.publish()
                self.pause = True
                return False, None

            if not pos_msg.success:
                rospy.logwarn('pausing -- send resume message when motion error is resolved')
                self.reset_pub.publish()
                self.pause_pub.publish()
                return False, None

        ### get latest data from subscriber
        data,robot_state,full_state,force,success = self._get_latest_data()
        
        if self.use_vel:
            self.robot.save_update(full_state,force=force,save=success)

        ### check if the robot moved
        if not success: 
            self.pause_pub.publish()
            self.pause = True
            return False, None

        if not self.states == self.states_orig:
            tmp_state = np.array(self.xinit_orig)
            tmp_state[self.explr_idx] = robot_state
            robot_state = tmp_state

        ### update_figs ###
        if self.use_vel:
            self.robot.check_plots()
            robot_data = self.robot.plot_data.copy()
            if isinstance(self.robot.plot_data[0],torch.Tensor):
                robot_data = [d.squeeze().detach().numpy() for d in robot_data]

            self.explr_update = [data.copy(),
                                robot_state.copy(),
                                force.copy(),
                                robot_data,
                                None, # zmu
                                None, # zvar
                                None, # img_pred
                                [iter_step,0]]


        return True, [robot_state,data,force]


class ExplrDist(object):
    def __init__(self,explr_idx=[0,1],capacity=1000):
        self.explr_idx = explr_idx
        self.capacity = capacity

        self.reset()

        self.init = False
        self.invert = False

    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        val = torch.ones(x.shape[0])
        val /= torch.sum(val)
        val += 1e-5
        return val
    
    def pdf_torch(self,samples,override_invert=False): 
        np_samples = samples.numpy()
        return torch.tensor(self.pdf(np_samples,override_invert),dtype=samples.dtype)

    def pdf(self,samples,override_invert=False):
        if (self.init and self.position > 0) or self.full_buffer:
            args = {}
            if self.full_buffer:
                means = self.env_path.copy()[:,self.explr_idx]
                stds = self.env_path_val.copy()
            else:
                means = self.env_path[:self.position].copy()[:,self.explr_idx]
                stds = self.env_path_val[:self.position].copy()
            dist = np.mean([mvn.pdf(samples, mean=mu, cov=std) for mu,std in zip(means,stds)],axis=0)
            if self.invert and not override_invert:
                dist = -dist+np.max(dist)+np.min(dist) # invert distribution
            return dist
        else:
            return self.init_uniform_grid(samples)

    def push(self,state,val):
        if (self.position + 1 ) == self.capacity:
            self.full_buffer = True
            print('buffer full')
        self.env_path[self.position] = state
        self.env_path_val[self.position] = val
        self.position = (self.position + 1) % self.capacity

    def reset(self):
        self.full_buffer = False
        self.position = 0
        self.env_path = np.empty([self.capacity,len(self.explr_idx)])
        self.env_path_val = np.empty([self.capacity,len(self.explr_idx)])
