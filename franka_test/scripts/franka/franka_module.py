#!/usr/bin/env python

########## global imports ##########
import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion, Point, WrenchStamped, TwistStamped
from sensor_msgs.msg import Image, CompressedImage, JointState
from franka_test.srv import GetStartState, UpdateVel, UpdateState, GetStartStateResponse, UpdateVelResponse, UpdateStateResponse
from franka_test.msg import BrightnessStamped
from std_msgs.msg import Empty
import tf

from termcolor import cprint
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
from argparse import Namespace
import yaml
from threading import Thread

########## local imports ##########
import sys, os
from franka.franka_env import FrankaEnv

class FrankaBridge(object):

    def __init__(self,args=None,rviz=False,node=False,use_thread=False):

        self.rviz=rviz
        self.use_thread = use_thread
        self.count = 0
        ## Initialize Franka Robot
        self.dt = rospy.get_param("dt", 0.5)
        start_x = rospy.get_param("start_x", 0.5)
        start_y = rospy.get_param("start_y", 0.)
        start_z = rospy.get_param("cam_z", 0.415)
        self.max_force = rospy.get_param("max_force", 500)
        sensor_method = rospy.get_param("sensor_method", "rgb")

        test_path = rospy.get_param('test_path', '')

        # path to saved vars
        base_path = rospy.get_param('base_path', './')

        if node:
            from load_config import get_config
            full_test_path = base_path + '/' + test_path + '/'

            # load variables
            config_loc = full_test_path + "/config.yaml"
            if os.path.exists(config_loc):
                # test config so load from previous training
                fingerprint_path = rospy.get_param("fingerprint_path", "eval/")
                fp_path = fingerprint_path.split(' ')[0] # just saving in the first place

                args = Namespace()
                with open(full_test_path + "/config.yaml","r") as f:
                    params = yaml.load(f, Loader=yaml.FullLoader)
                for k, v in params.items():
                    setattr(args, k, v)

                # load exploration / training config modification (if applicable)
                test_config_file = rospy.get_param("test_config_file", "fp_trainer_config.yaml")
                save_name = rospy.get_param('save_name', 'test')

                src_file = base_path+'/config/' + test_config_file
                with open(src_file) as f:
                    tmp = yaml.load(f,Loader=yaml.FullLoader)
                for _, param in tmp.items(): # top level is for user readability
                    for k, v in param.items():
                        setattr(args, k, v)
            else: 
                # new config so load from scratch file
                args = get_config(print_output=False)

        render = rospy.get_param("render_pybullet", True)
        env_args = {}
        if 'x' in args.states: 
            x_idx = args.states.rfind('x')
            env_args['xlim'] = args.tray_lim[x_idx]
        if 'y' in args.states: 
            y_idx = args.states.rfind('y')
            env_args['ylim'] = args.tray_lim[y_idx]
        if np.all([s in args.states for s in 'xyzrpw']):
            self.full_control = True
            self.level_ee = False
            self.fix_z = False
        else: 
            self.full_control = False
            self.level_ee = not('r' in args.states) and not('p' in args.states)
            self.fix_z = not ('z' in args.states)
        env_args['offset']=[0.,0.,0.]
        self.env = FrankaEnv(render=render, ts=self.dt, img_shape=args.raw_image_dim[:2],sensor_method=sensor_method,base_path=base_path+'/scripts/franka/',**env_args)

        self.start_pos = np.array([start_x, start_y, start_z])
        self.start_rpw = np.array([np.pi, 0., -np.pi/8])
        self.start_orn = Rotation.from_euler('xyz',self.start_rpw).as_quat()
        for _ in range(10):
            self.env.step(self.start_pos, self.start_orn)
        # self.cmd = [None]*4
        self.update_pybullet_thread = None

        if self.env.use_camera:
            current_brightness = 1. # starting
            self.max_brightness = 1. # limits
            self.env.update_brightness(min(current_brightness,self.max_brightness)) # reset

        if self.rviz:
            self.img_raw_pub = rospy.Publisher('/usb_cam/image_raw/compressed',CompressedImage,queue_size=1,latch=False)
            self.joint_state_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)

        if node: 
            ### Setup Ros Env ###
            rospy.Subscriber('/update_processes',Empty,self.update_process_callback)
            self.img_pub = rospy.Publisher('/usb_cam/image',Image,queue_size=1,latch=False)
            self.vel_pub = rospy.Publisher('/ee_vel',TwistStamped,queue_size=1,latch=False)
            self.force_pub = rospy.Publisher('/ee_wrench',WrenchStamped,queue_size=1,latch=False)

            if self.env.use_camera:
                self.brightness_pub = rospy.Publisher('/usb_cam/brightness',BrightnessStamped,queue_size=1,latch=False)

            rospy.Timer(rospy.Duration(1/30.), self.publishStates)

        self.pose_pub = rospy.Publisher('/ee_pose',PoseStamped,queue_size=1,latch=False) # used by gui
        rospy.Timer(rospy.Duration(1/30.), self.publishPose)
        rospy.Service('/klerg_start_pose',GetStartState,self.startCallback)
        rospy.Service('/klerg_cmd',UpdateVel,self.velCallback)
        rospy.Service('/klerg_pose',UpdateState,self.poseCallback)
        rospy.Subscriber('/reset',Empty,self.resetCallback)
        rospy.Subscriber('/reset_joints',Empty, self.resetJointsCallback)
        rospy.Subscriber('/move_objects',Empty,self.moveObjectsCallback)
        rospy.Subscriber('/add_object',Empty,self.addObjectCallback)
        cprint('[pybullet] setup complete','cyan')

    def update_process_callback(self,msg):
        import psutil
        p = psutil.Process()
        node_name = rospy.get_name()
        ros_affinity = rospy.get_param('ros_affinity',{})
        if node_name in ros_affinity.keys():
            cores = ros_affinity[node_name]
            p.cpu_affinity(cores)

    def get_state(self): 

        # Get ee force (inverted from fts)
        force_msg = WrenchStamped()
        force_msg.wrench.force = Point(*self.env.curr_force[:3].copy())
        force_msg.wrench.torque = Point(*self.env.curr_force[3:].copy())

        vel_msg = TwistStamped()
        vel_msg.twist.linear = Point(*self.env.curr_lin_vel.copy())
        vel_msg.twist.angular = Point(*self.env.curr_ang_vel.copy())

        # Get sensor image
        img = np.array(self.env.cam_img[:,:,:3])

        # Send sensor image
        img = np.transpose(img,[1,0,2])
        img_msg = Image()
        img_msg.height = img.shape[0]
        img_msg.width = img.shape[1]
        img_msg.data = img.tobytes()

        brightness = self.env.current_brightness
        
        pose_msg = PoseStamped()
        pose_msg.pose.position = Point(*self.env.curr_pos.copy())
        pose_msg.pose.orientation = Quaternion(*self.env.curr_orn.copy())

        if self.rviz:
            self.publishCompressedImg(img)

        return pose_msg,vel_msg,force_msg,brightness,img_msg

    def publishCompressedImg(self,img):
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        imageBRG = cv2.cvtColor(img , cv2.COLOR_RGB2BGR)
        msg.data = np.array(cv2.imencode('.jpg', imageBRG)[1]).tobytes()
        self.img_raw_pub.publish(msg)

    def publishStates(self,event): 
        self.publishPose(None)
        self.publishImage(None)
        self.publishVel(None)
        self.publishForce(None)
        if self.env.use_camera:
            self.publishBrightness(None)

    def publishJointState(self,event):
        joint_states = self.env.get_joint_states()
        joint_msg = JointState()
        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.name = [f'panda_joint{link+1}' for link in range(len(joint_states))]
        joint_msg.position = [state[0] for state in joint_states]
        # joint_msg.velocity = [state[1] for state in joint_states]
        # joint_msg.effort = [state[3] for state in joint_states]
        self.joint_state_pub.publish(joint_msg)

    def publishForce(self,event):
        # Get ee force (inverted from fts)
        msg = WrenchStamped()
        msg.wrench.force = Point(*self.env.curr_force[:3].copy())
        msg.wrench.torque = Point(*self.env.curr_force[3:].copy())
        msg.header.frame_id = 'ee_frame'
        msg.header.stamp = rospy.Time.now()
        self.force_pub.publish(msg)
        rospy.logwarn_once('[pybullet] published ee_force')

    def publishVel(self,event):
        msg = TwistStamped()
        msg.twist.linear = Point(*self.env.curr_lin_vel.copy())
        msg.twist.angular = Point(*self.env.curr_ang_vel.copy())
        msg.header.frame_id = 'ee_frame'
        msg.header.stamp = rospy.Time.now()
        self.vel_pub.publish(msg)
        rospy.logwarn_once('[pybullet] published ee_vel')

    def publishImage(self,event):
        # Get sensor image
        img = np.array(self.env.cam_img[:,:,:3])
        # Send sensor image
        img = np.transpose(img,[1,0,2])
        msg = Image()
        msg.header.stamp = rospy.Time.now()
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.data = img.tobytes()
        self.img_pub.publish(msg)
        if self.rviz:
            self.publishCompressedImg(img)
        rospy.logwarn_once('[pybullet] published image message')

    def publishPose(self,event):
        msg = PoseStamped()
        msg.pose.position = Point(*self.env.curr_pos.copy())
        msg.pose.orientation = Quaternion(*self.env.curr_orn.copy())
        msg.header.frame_id = 'world'
        msg.header.stamp = rospy.Time.now()
        self.pose_pub.publish(msg)
        rospy.logwarn_once('[pybullet] published ee_pose')

        if self.rviz:
            br = tf.TransformBroadcaster()
            br.sendTransform(self.env.curr_pos.copy(),
                            tf.transformations.quaternion_from_euler(0, 0, 0),
                            rospy.Time.now(),
                            "ee_frame","world")
            self.publishJointState(None)

    def moveObjectsCallback(self,event):
        self.env.move_objects()

    def addObjectCallback(self,event):
        self.env.add_object()

    def publishBrightness(self,event):
        msg = BrightnessStamped()
        msg.brightness = self.env.current_brightness
        msg.header.frame_id = 'usb_cam'
        msg.header.stamp = rospy.Time.now()
        self.brightness_pub.publish(msg)

    def velCallback(self,msg):
        self.count += 1
        # Generate target position in env
        d_pos = np.array([msg.desired_vel.linear.x, msg.desired_vel.linear.y, msg.desired_vel.linear.z])
        d_orn = np.array([msg.desired_vel.angular.x, msg.desired_vel.angular.y, msg.desired_vel.angular.z]) #*15.
        brightness = min(msg.desired_brightness,self.max_brightness)

        # check force 
        force = self.env.curr_force[:3].copy()
        norm_force = np.linalg.norm(force)
        if norm_force > 0.75*self.max_force: 
            d_pos[np.sign(force) != np.sign(d_pos)] = 0. # move away from force
            d_orn[:] = 0. # stop rotating

        cmd = ['vel',d_pos,d_orn,brightness]

        rospy.logwarn_once('[pybullet] updated pybullet')
        rospy.logwarn_once('[pybullet] sending robot pose')
        pose = Pose()
        pose.position = Point(*self.env.curr_pos.copy())
        pose.orientation = Quaternion(*self.env.curr_orn.copy())
        if self.use_thread:
            if self.update_pybullet_thread is not None: 
                self.update_pybullet_thread.join()
            self.update_pybullet_thread = Thread(target=self.updatePybullet,args=(cmd,))
            self.update_pybullet_thread.start()
        else:
            self.updatePybullet(cmd)
        return UpdateVelResponse(pose,True)

    def updatePybullet(self,cmd):
        move,pos,orn,brightness = cmd
        # move,pos,orn,brightness = self.cmd
        if brightness > -1:
            self.env.update_brightness(brightness)
        if move == 'vel':
            # apply velocity
            self.env.step(pos, orn, use_vel = True, leave_trace=True)

            # occasionally check for drift if not controlling pitch/yaw (or pitch/yaw/z)
            if (self.count % 20 == 0) and not(self.full_control): 
                fix_pose = self.env.curr_pos.copy()
                fix_orn = self.env.curr_orn.copy()
                if self.fix_z: 
                    fix_pose[2] = self.start_pos.copy()[2]
                if self.level_ee: 
                    fix_rot = Rotation.from_quat(fix_orn).as_euler('xyz')
                    fix_rot[0] = self.start_rpw.copy()[0]
                    fix_rot[1] = self.start_rpw.copy()[1]
                    fix_orn = Rotation.from_euler('xyz',fix_rot).as_quat()
                self.env.step(fix_pose, fix_orn, save_update=False)
        elif move == 'pose':
            self.env.step(pos, orn)
        # self.cmd = [None]*4

    def poseCallback(self,msg):
        # Generate target position in env
        pos = np.array([msg.desired_pose.position.x, msg.desired_pose.position.y, msg.desired_pose.position.z])
        orn = np.array([msg.desired_pose.orientation.x, msg.desired_pose.orientation.y,
                        msg.desired_pose.orientation.z,msg.desired_pose.orientation.w])
        brightness = min(msg.desired_brightness,self.max_brightness)

        cmd = ['pose',pos,orn,brightness]
        rospy.logwarn_once('[pybullet] updated pybullet')
        rospy.logwarn_once('[pybullet] sending robot pose')
        pose = Pose()
        pose.position = Point(*self.env.curr_pos.copy())
        pose.orientation = Quaternion(*self.env.curr_orn.copy())
        if self.use_thread:
            if self.update_pybullet_thread is not None: 
                self.update_pybullet_thread.join()
            self.update_pybullet_thread = Thread(target=self.updatePybullet,args=(cmd,))
            self.update_pybullet_thread.start()
        else:
            self.updatePybullet(cmd)
        return UpdateStateResponse(pose,True)

    def startCallback(self,msg):
        rospy.logwarn_once('[pybullet] sending start pose')
        pose = Pose()
        pose.position = Point(*self.env.curr_pos.copy())
        pose.orientation = Quaternion(*self.env.curr_orn.copy())
        return GetStartStateResponse(pose,True)

    def resetCallback(self,msg):
        for _ in range(10):
            self.env.step(self.start_pos, self.start_orn)
        
    def resetJointsCallback(self,msg):
        self.env.reset()

