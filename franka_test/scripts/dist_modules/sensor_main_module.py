#!/usr/bin/env python

########## global imports ##########
import warnings
warnings.filterwarnings('ignore')
import torch
import numpy as np
import pickle
import time
import copy

from termcolor import cprint

import rospy
from std_msgs.msg import Empty
from franka_test.srv import *
from franka_test.msg import Distribution

########## local imports ##########
import sys, os
from franka.franka_utils import ws_conversion
from .sensor_utils import SensorMainRosBase


class SensorMain(SensorMainRosBase):
    def __init__(self, model, optimizer, replay_buffer, args, killer, explr_robot_lim_scale = 1.15, shared_model = None):
        self.model = model
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.killer = killer
        self.map_dict(vars(args))

        self.vae_buffer = replay_buffer

        # initalize buffers
        self.tray_lim = np.array(self.tray_lim)
        self.robot_lim = np.array(self.robot_lim)
        self.tray_ctrl_lim = np.array(self.tray_ctrl_lim)
        self.robot_ctrl_lim = np.array(self.robot_ctrl_lim)
        self.explr_robot_lim_scale = explr_robot_lim_scale
        self.path = [] # np.empty(self.num_steps,dtype=object)
        self.env_path = [] # np.empty(self.num_steps,dtype=object)
        self.actions = [] # np.empty(self.num_steps,dtype=object)
        self.data_buffer = [] # np.empty(self.num_steps,dtype=object)
        self.iter_step = 0
        self.learning_ind = 0
        self.fname = None

        self.explr_update = None
        self.cam_img = None
        self.ee_pose = None
        self.robot = None
        self.xinit = None
        self.got_state = False

        if self.explr_method == 'uniform':
            self.use_vel = False

        # ros
        super(SensorMain, self).__init__(self.tray_lim, self.robot_lim,self.dir_path,self.states,self.plot_states)
        self.data_to_ctrl_rate = rospy.get_param('/data_to_ctrl_rate',1)
        self.pybullet = rospy.get_param('pybullet', False)
        self.dist_pub = rospy.Publisher('/clustering/distribution',Distribution,queue_size=10,latch=False)
        self.publish_distribution = False
        if not self.pybullet: 
            if self.use_vel:
                rospy.wait_for_service('klerg_cmd')
                self.send_cmd = rospy.ServiceProxy('/klerg_cmd', UpdateVel)
            else:
                rospy.wait_for_service('klerg_pose')
                self.send_cmd = rospy.ServiceProxy('/klerg_pose', UpdateState)
        else: 
            if self.use_vel:
                self.send_cmd = lambda a,b: self.franka_env.velCallback(UpdateVelRequest(a,b))
            else: 
                self.send_cmd = lambda a,b: self.franka_env.poseCallback(UpdateStateRequest(a,b))
        while self.xinit is None:
            self.rate.sleep()
        self.start_robot()
        if not self.use_vel:
            start_pose_controller = rospy.Publisher("/switch_to_pose_controller",Empty,queue_size=1,latch=True)
            rospy.logwarn("switching controller")
            start_pose_controller.publish()
            rospy.wait_for_service('/klerg_pose')
        time.sleep(0.5)
        rospy.loginfo_once("ready to run")

    def start_robot(self):
        # Initialize KL-Erg Robot
        if 'klerg' in self.explr_method:
            from control_torch.klerg import Robot
        else:
            from control.dummy_robot import DummyRobot as Robot

        self.explr_idx = list(range(len(self.states)))

        args = {}
        args['x0'] = np.hstack([*self.xinit, *np.zeros(len(self.xinit))])
        if self.ddp_model:
            args['target_dist'] = self.model.module
        else:
            args['target_dist'] = self.model

        shared_vars = ['vel_states','states','dt','explr_idx','horizon','tray_lim','robot_ctrl_lim','R','plot_states','explr_robot_lim_scale','use_magnitude','use_vel','std','pybullet']
        for key in shared_vars:
            args[key] = getattr(self,key)

        self.robot = Robot(robot_lim=self.robot_lim.copy(), 
                            buffer_capacity=self.traj_buffer_capacity,
                            plot_data=True, uniform_tdist=('unif' in self.explr_method),**args)

    def extra_image_processing(self,data):
        # placeholder for possible downsampling later
        return data

    # @torch.no_grad()
    def step(self,iter_step,move_only=False):
        if iter_step == self.prior_steps: 
            self.robot.use_prior = False
        self.model.eval()
        if iter_step % self.data_to_ctrl_rate == 0: # update control less frequently
            ## update simulated robot to current pose
            if not self.pybullet:
                pos,full_pos,force = self.get_latest_pose()
                if pos is None: # error handling if data isn't available
                    return False, None 
                full_state = ws_conversion(full_pos, self.tray_full_lim, self.robot_full_lim) # double integrator format (w/vel)
                self.robot.save_update(full_state,force=force,save=False)

            ### step robot ###
            state, vel, action = self.robot.step(num_target_samples=self.num_target_samples, num_traj_samples=self.num_traj_samples,save_update=False)

            # error handling
            if np.any(np.isnan(state)):
                rospy.logwarn('got nan in state')
                return False, None

            ### Step in Franka environment ###
            # Convert klerg state to environment state
            tray_pos = ws_conversion(state, self.robot_lim, self.tray_lim)
            # Generate target position in env
            if self.use_vel:
                vel = ws_conversion(vel, self.robot_ctrl_lim, self.tray_ctrl_lim)
                vel = np.clip(vel,*np.array(self.tray_ctrl_lim).T)
                cmd = self.format_Twist_msg(vel)
            else:
                cmd = self.format_Pose_msg(tray_pos)
            if self.brightness_idx >= 0: 
                brightness = tray_pos[self.brightness_idx]
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

            if (self.explr_method == 'uniform'):
                # rospy.sleep(0.5)
                at_center = self.check_goal_pos(tray_pos,brightness)
                if not at_center: 
                    self.write_to_log(f"didn't make it to goal pose step {iter_step}")
                if not self.pybullet:
                    self.resume_callback(None)
        else: 
            state = None
            action = None

        if move_only:
            return False, None

        ### get latest data from subscriber
        data,pos,full_pos,force,data_success = self.get_latest_msg()
        ### check if the robot moved
        data_success = self.check_cmd(pos) and data_success

        # Convert env state to klerg state
        robot_state = ws_conversion(pos, self.tray_lim, self.robot_lim)
        full_state = ws_conversion(full_pos, self.tray_full_lim, self.robot_full_lim) # double integrator format
        self.robot.save_update(full_state,force=force,save=data_success)
        if self.robot.use_magnitude: 
            robot_state[self.robot.vel_locs] = np.abs(robot_state[self.robot.vel_locs])

        if not data_success: # data out of sync
            return False, None

        ### Update data buffer ###
        self.env_path.append(pos) # self.env_path[iter_step] = pos #

        ### Update lists ###
        self.path.append(robot_state) # self.path[iter_step] = robot_state
        self.actions.append(action) # self.actions[iter_step] = action

        # Push to buffer (numpy)
        self.data_buffer.append([robot_state.copy(),data.T.copy()]) # self.data_buffer[iter_step] = [robot_state,data.copy()]
        data = self.extra_image_processing(data).T
        with torch.no_grad():
            # Push to buffer (torch)
            robot_state = torch.as_tensor(robot_state).to(device=self.device,dtype=self.dtype)
            data = torch.as_tensor(data).to(device=self.device,dtype=self.dtype) # ,memory_format=torch.channels_last)
            force = torch.as_tensor(force).to(device=self.device,dtype=self.dtype)

            self.vae_buffer.push(robot_state,data,force=force)

            self.robot.check_plots()
            x_r = robot_state.unsqueeze(axis=0)
            y_r = data.unsqueeze(axis=0)
            force_r = force.unsqueeze(axis=0)

            # before iter:
            if self.learn_force:
                out = self.model(x_r, y_r,force_r)
            else:
                out = self.model(x_r, y_r)
            pre_iter_img_pred, z_mu, z_var = out[0],out[2],out[3]

            # update_figs
            self.explr_update = [data.detach().clone().permute(2,1,0),
                                robot_state.detach().clone(),
                                force.detach().clone(),
                                self.robot.plot_data.copy(),
                                z_mu.detach().clone(),
                                z_var.exp().detach().clone(),
                                pre_iter_img_pred[0].detach().clone().permute(2,1,0),
                                [iter_step,self.learning_ind]]

            if self.publish_distribution:
                dist_msg = Distribution()
                samples=self.robot.plot_data[0]
                target_dist=self.robot.plot_data[4] # 1 for raw, 4 for smoothed
                dist_msg.explr_step = iter_step
                dist_msg.learning_ind = self.learning_ind
                dist_msg.samples_layout = samples.shape
                dist_msg.samples = samples.flatten()
                dist_msg.distribution = target_dist
                self.dist_pub.publish(dist_msg)
                self.publish_distribution = False

        return True,[robot_state,data]

    @torch.no_grad()
    def post_train_mp(self,iter_step,learning_ind=None): # only for unif klerg and random sampling
        self.iter_step = iter_step
        if learning_ind is not None:
            self.learning_ind = learning_ind

        # Update KL-Erg Target distribution
        if (iter_step % self.update_rate == 0) and (iter_step > self.frames_before_update):
            buff = self.vae_buffer.get_last()
            if self.learn_force:
                inputs = buff[:3]
            else:
                inputs = buff[:2]
            if self.ddp_model:
                self.model.module.update_dist(*inputs)
            else:
                self.model.update_dist(*inputs)

    @torch.no_grad()
    def save(self,post_explr=False,callback=False,mod="",losses='None',save_model=True):
        # Save Pickled Data
        data_eval_dict = {
            "path": np.array(self.path),
            "actions": np.array(self.actions),
            # "buffer": self.data_buffer, # moved to a different location to make file smaller
            "env_path": np.array(self.env_path),
            "losses": losses,
            "tray_lim": self.tray_lim,
            "klerg_lim": self.robot_lim,
            "learning_ind" : self.learning_ind,
            "iter_step": self.iter_step,
            "states": self.states
            }

        with open( self.dir_path+"data_eval_dict_explr.pickle", "wb" )  as f:
            pickle.dump(data_eval_dict,f,protocol=pickle.HIGHEST_PROTOCOL)

        # Save Torch final model
        if save_model: 
            if mod == "":
                if post_explr:
                    mod = "_postexplr"
                    self.write_to_log(f'final runtime: {self.duration_str}')
                elif callback:
                    mod = f"_{self.iter_step}steps_callback"
                    self.write_to_log(f'callback runtime: {self.duration_str}')
                else:
                    self.write_to_log(f'explr_learning runtime: {self.duration_str}')

            if self.ddp_model:
                torch.save(self.model.module.state_dict(), self.dir_path+'model_final'+mod+'.pth') # state dict only
                # torch.save(self.model.module, self.dir_path+'model_final'+mod+'.pth') # full model requires path continuity
            else:
                torch.save(self.model.state_dict(), self.dir_path+'model_final'+mod+'.pth')  # state dict only
                # torch.save(self.model, self.dir_path+'model_final'+mod+'.pth') # full model requires path continuity
            # if self.optimizer is not None:
            #     torch.save(self.optimizer.state_dict(), self.dir_path+'optim_final'+mod+'.pth') # state dict only
                # torch.save(self.optimizer, self.dir_path+'optim_final'+mod+'.pth') # full model requires path continuity

    @torch.no_grad()
    def load_model(self,iter_step=None,pre_iter_img_pred=None,shared_model=False):
        PATH=self.dir_path+'model_checkpoint_tmp.pth'
        if not(shared_model) and os.path.exists(self.dir_path+'model_ready'):
            try:
                tmp = torch.load(PATH)
                self.model.load_state_dict(tmp['model'],strict=False)
                self.learning_ind = tmp['epoch']
                # print('loaded model',self.learning_ind)
                # os.remove(PATH)
                os.remove(self.dir_path+'model_ready')
                if self.print_debug: cprint('loaded model','green')
                # losses = None
                # start_ind = self.learning_ind
                # self.learning_ind += self.num_learning_opt
                # self.post_train_mp(iter_step,losses,pre_iter_img_pred=pre_iter_img_pred,start_ind=start_ind)
            except:
                pass
        elif shared_model and not (self.learning_ind == self.shared_model.learning_ind.item()):
            try:
                # Update KL-Erg Target distribution
                if self.model.init:
                    buff = self.vae_buffer.get_last()
                    if self.learn_force:
                        inputs = buff[:3]
                    else:
                        inputs = buff[:2]
                    self.shared_model.update_dist(*inputs)
                self.model.load_state_dict(copy.deepcopy(self.shared_model.state_dict()),strict=False)
                self.learning_ind = self.shared_model.learning_ind.item()
            except: 
                pass

    @torch.no_grad()
    def save_clustering_model(self,shared_model=False):
        if shared_model: 
            if not self.async_trainer: 
                self.shared_model.learning_ind[0] = self.learning_ind
                self.shared_model.load_state_dict(copy.deepcopy(self.model.module.state_dict()) if self.ddp_model else copy.deepcopy(self.model.state_dict()),strict=False)
        else:
            tmp = {'model':self.model.module.state_dict() if self.ddp_model else self.model.state_dict(),
            'epoch':self.learning_ind}
            torch.save(tmp, self.dir_path+'clustering_model_checkpoint_tmp.pth')
            with open(self.dir_path+'clustering_model_ready','w') as f:
                pass

