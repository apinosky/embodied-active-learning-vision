#!/usr/bin/env python

########## global imports ##########
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import numpy.random as npr
import time
import datetime
import copy

import rospy
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import Pose, Twist, PoseStamped, WrenchStamped, TwistStamped, Point, Quaternion, Transform
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, ColorRGBA, Header
from franka_test.srv import GetStartState, PoseCmd, UpdateVel
from franka_test.msg import BrightnessStamped
from franka_msgs.msg import ErrorRecoveryActionGoal
import message_filters
from visualization_msgs.msg import Marker
from std_srvs.srv import Empty as EmptySrv
from std_srvs.srv import EmptyResponse as EmptySrvResponse

########## local imports ##########
from franka.franka_utils import ws_conversion


def numba_handler(num, stack):
    raise Exception("numba stuck")

def get_cost(d):
    robot_data = d[3]
    return robot_data[-1]

def get_circular_traj(radius=0.25,num_points=10,center=np.array([0,0])):
    def get_point(radius, angle, center):
        return center + radius*np.array([np.cos(angle),np.sin(angle)])
    start = npr.choice(num_points)
    step = 2*np.pi / num_points
    traj = np.stack([get_point(radius,(start + i)*step,center) for i in range(num_points)])
    return traj

def get_grid_traj(grid_size=0.1, num_points=10,center=np.array([0,0])):
    mul = np.round(np.sqrt(num_points))
    x = np.linspace(-1, 1, int(mul))
    y = np.linspace(-1, 1, int(mul))
    # full coordinate arrays
    xx, yy = np.meshgrid(x, y)
    # resize to desired grid
    xx *= grid_size/2
    yy *= grid_size/2
    # offset to desired center
    xx += center[0]
    yy += center[1]
    # flip even rows to make connected path
    xx[::2] = np.fliplr(xx[::2]) # flip even rows
    traj = np.array([np.ravel(xx),np.ravel(yy)]).T
    return traj

def from_vector3(msg):
    return np.array([msg.x, msg.y, msg.z])

def from_vector4(msg):
    return np.array([msg.x, msg.y, msg.z, msg.w])

class SensorMainRosBase(object):
    def __init__(self, tray_lim, robot_lim, dir_path, states='xy',plot_states='xy'):

        self.tray_lim = tray_lim
        self.robot_lim = robot_lim
        self.dir_path = dir_path
        self.log_file = "log.txt"

        self.update_states(states,plot_states)

        self.cam_img = None
        self.ee_pose = None
        self.xinit = None
        self.pos_init = None
        self.rot_init = None
        self.last_state = None
        self.got_state = False
        self.got_img = False
        self.pause = False
        self.manual = False
        self.start_time = time.time()

        # ros
        rospy.init_node('cam_explore')
        dt = rospy.get_param('/dt',0.2)
        self.use_pybullet = rospy.get_param('/pybullet',False)
        if self.use_pybullet:
            from franka.franka_module import FrankaBridge
            self.franka_env = FrankaBridge(args=self)
            self.got_img = True
            dt /= 5.
            print('increasing simulation speed')
        data_to_ctrl_rate = rospy.get_param('/data_to_ctrl_rate',1)
        self.rate = rospy.Rate(1./dt*data_to_ctrl_rate)
        # tell robot whether or not to apply force limits
        rospy.set_param('/max_force',self.max_force)
        rospy.set_param('/enforce_dt',True)
        # move end effector frame (if applicable) -- must do this before moving!
        rospy.set_param('/sensor_mass',self.sensor_mass)
        if (self.sensor_z_offset is not None) and (not self.use_pybullet): 
            rospy.wait_for_service('update_ee_frame')
            update_ee_offset = rospy.ServiceProxy('/update_ee_frame', PoseCmd)
            ee_offset = Transform()
            ee_offset.translation.z = self.sensor_z_offset
            ee_offset.rotation.w = 1.
            update_ee_offset(ee_offset)
        if not self.use_pybullet:
            # update robot z to match cam_z
            set_start = rospy.Publisher('/set_new_reset_xyz',Point,queue_size=1,latch=True)
            msg = Point(1000,1000,self.cam_z) # set >=100 to ignore dimension
            set_start.publish(msg)
            time.sleep(5.)

        # then set up other publishers
        self.stop_pub = rospy.Publisher('/stop',Empty,queue_size=1,latch=True)
        self.reset_pub = rospy.Publisher('/franka_control/error_recovery/goal',ErrorRecoveryActionGoal,queue_size=1,latch=True)
        self.reset_joints_pub = rospy.Publisher('/reset_joints',Empty,queue_size=1)
        self.reset_ctrls_pub = rospy.Publisher('/reset_control_commands',Empty,queue_size=1)
        self.marker_pub = rospy.Publisher('/projection', Marker, queue_size=10)
        self.pause_pub = rospy.Publisher('/pause',Empty,queue_size=1)
        self.marker_count = 0
        rospy.Subscriber('/manual',Empty,self.manual_callback)
        rospy.Subscriber('/disable_manual',Empty,self.disable_manual_callback)
        rospy.Subscriber('/pause',Empty,self.pause_callback)
        rospy.Subscriber('/resume',Empty,self.resume_callback)
        rospy.Subscriber('/save',Empty,self.save_callback)
        rospy.wait_for_service('klerg_start_pose')
        start_service = rospy.ServiceProxy('/klerg_start_pose', GetStartState)
        self.start_callback(start_service())
        if not self.use_pybullet:
            rospy.Subscriber('/usb_cam/image',Image,self.image_callback)
            brightness_sub = message_filters.Subscriber('/usb_cam/brightness',BrightnessStamped) # @ 30 Hz like img
            self.brightness_cache = message_filters.Cache(brightness_sub, 10)
            # image_sub = message_filters.Subscriber('/usb_cam/image',Image)
            pose_sub = message_filters.Subscriber('/ee_pose',PoseStamped) # note: img @ 30Hz, others @ 100Hz
            self.pose_cache = message_filters.Cache(pose_sub, 10)
            # joint_sub = message_filters.Subscriber('/joint_states',JointState)
            # self.joint_cache = message_filters.Cache(joint_sub, 10)
            # self.combo_pub = rospy.Publisher('/combo',Empty,queue_size=1)
            force_sub = message_filters.Subscriber('/ee_wrench',WrenchStamped)
            self.force_cache = message_filters.Cache(force_sub, 10)
            vel_sub = message_filters.Subscriber('/ee_vel',TwistStamped)
            self.vel_cache = message_filters.Cache(vel_sub, 10)
            # ts = message_filters.ApproximateTimeSynchronizer([image_sub,pose_sub],queue_size=1,slop=0.005)
            # ts.registerCallback(self.poseANDimage_callback)
        rospy.Subscriber('/move_objects',Empty,self.resetHistoryCallback)

        affinity_ready = rospy.Publisher('/update_processes',Empty,queue_size=1,latch=True)
        affinity_ready.publish()

        rospy.Service('/sensor_utils',EmptySrv, self.startupCallback)

    def startupCallback(self,msg): 
        return EmptySrvResponse()

    def map_dict(self, user_info):
        for k, v in user_info.items():
            setattr(self, k, v)

    def update_states(self,states,plot_states='xy'):
        state_dict = {}
        last_lower = 0
        for idx,s in enumerate('xyzrpwbXYZRPWB'):
            state_dict[s] = idx
            if s.lower() == s:
                last_lower = idx
        out = []
        non_vel_states = []
        non_vel_idx = []
        vel_states = []
        for state_loc,key in enumerate(states):
            idx = state_dict[key]
            out.append(idx)
            if idx <= last_lower:
                non_vel_states.append(state_dict[key])
                vel_states.append(state_dict[key.upper()])
                non_vel_idx.append(state_loc)
        self.states = states
        self.plot_idx = [self.states.rfind(s) for s in plot_states]
        self.msg_states = out
        self.full_msg_states = non_vel_states + vel_states
        # self.non_vel_states = non_vel_states
        # self.vel_states = vel_states
        self.robot_full_lim = np.vstack([self.robot_lim[non_vel_idx],self.robot_ctrl_lim[non_vel_idx]])
        self.tray_full_lim = np.vstack([self.tray_lim[non_vel_idx],self.tray_ctrl_lim[non_vel_idx]])
        self.brightness_idx = self.states.rfind('b')
        print(states,out)
        return non_vel_idx

    @property
    def duration_str(self):
        return str(datetime.timedelta(seconds=(time.time()-self.start_time)))

    def write_to_log(self,msg):
        print(msg)
        with open(self.dir_path + self.log_file,"a") as f:
            f.write(msg+'\n')

    def format_Twist_msg(self,vel):
        cmd = Twist()
        for val,key in zip(vel,self.states):
            if key == 'x':
                cmd.linear.x = val
            elif key == 'y':
                cmd.linear.y = val
            elif key == 'z':
                cmd.linear.z = val
            elif key == 'r':
                cmd.angular.x = val
            elif key == 'p':
                cmd.angular.y = val
            elif key == 'w':
                cmd.angular.z = val
        return cmd

    def format_Pose_msg(self,pose):
        cmd = Pose()
        cmd.position = copy.copy(self.pos_init)
        rot = self.rot_init.copy()
        for val,key in zip(pose,self.states):
            if key == 'x':
                cmd.position.x = val
            elif key == 'y':
                cmd.position.y = val
            elif key == 'z':
                cmd.position.z = val
            elif key == 'r':
                rot[0] = val
            elif key == 'p':
                rot[1] = val
            elif key == 'w':
                rot[2] = val
        quat = Rotation.from_euler('xyz',rot).as_quat()
        cmd.orientation = Quaternion(*quat)
        # error handling
        quat_msg = cmd.orientation
        quat = np.array([quat_msg.x , quat_msg.y, quat_msg.z, quat_msg.w])
        valid = np.abs(1-np.linalg.norm(quat))<0.001
        if not valid:
            rospy.logerr('invalid quaternion')

        return cmd


    # https://www.ros.org/reps/rep-0103.html#rotation-representation
    # see link above for refernce of rotation representation for angular velocities
    # (fixed axis roll, pitch, yaw about X, Y, Z axes respectively)
    def process_pose_msg(self,pose,vel=None,brightness=None,init=False):

        pos = from_vector3(pose.position)
        quat = from_vector4(pose.orientation)

        rot = Rotation.from_quat(quat).as_euler('xyz')
        rot[0] = rot[0] % (2 * np.pi) # wrap btwn 0 and 2*pi
        rot[1:] = ((rot[1:] + np.pi) % (2 * np.pi)) - np.pi # wrap btwn -pi and pi

        if init:
            self.rot_init = rot.copy()
            self.pos_init = copy.copy(pose.position)

        self.updateMarker(pos,rot)
        # print(np.round(rot[3:6],3))

        if vel is None:
            lin_vel = np.zeros(3)
            ang_vel = np.zeros(3)
        else:
            lin_vel = from_vector3(vel.linear)   # NOTE: scaling may be off
            ang_vel = from_vector3(vel.angular)  # NOTE: scaling may be off

        if brightness is None:
            brightness = 0.5
        brightness_vel = 0.

        states = np.hstack([pos,rot,brightness,lin_vel,ang_vel,brightness_vel])
        # print(states)
        return states[self.msg_states], states[self.full_msg_states]

    def process_image_msg(self,image_msg,make_square=True):
        tmp = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(image_msg.height, image_msg.width, -1)
        # tmp = np.flipud(tmp)
        if make_square and image_msg.width > image_msg.height: # make image square
            offset = int((image_msg.width-image_msg.height)/2)
            tmp = tmp[:, offset:-offset,:]
        if self.zoom > 1: 
            offset = int(image_msg.height/2 - image_msg.height/self.zoom/2)
            tmp = tmp[offset:-offset, offset:-offset,:]
        tmp = tmp[::self.down_sample,::self.down_sample,:]
        tmp = tmp/255.0
        if self.intensity:
            tmp = np.mean(tmp, axis=2, keepdims=True)
        return tmp


    def start_callback(self,msg):
        rospy.loginfo_once("got start pose")
        # joint_msg = rospy.wait_for_message('/joint_states',JointState)
        # self.ee_loc = np.argwhere([x=='panda_joint7' for x in joint_msg.name]).squeeze()
        # ee_state = joint_msg.position[self.ee_loc]
        start_vel = Twist() # init with zeros
        pos,full_pos = self.process_pose_msg(msg.start_pose,start_vel,init=True)
        self.xinit = ws_conversion(np.array(pos), self.tray_lim, self.robot_lim)
        self.got_state = True

    def poseANDimage_callback(self,image_msg,pose_msg):
        rospy.loginfo_once("got image")
        self.cam_img = image_msg
        self.ee_pose = pose_msg.pose
        self.got_img = True
        # self.combo_pub.publish()

    def get_latest_pose(self):
        if self.use_pybullet: 
            pose_msg,vel_msg,force_msg,brightness,img_msg=self.franka_env.get_state()
        else:
            ## get most recent pose
            time_stamp = self.pose_cache.getLatestTime()
            if time_stamp is None: 
                return [None,None,None]
            closest_poses = [self.pose_cache.getElemBeforeTime(time_stamp),
                            self.pose_cache.getElemAfterTime(time_stamp)]
            pose_msg = closest_poses[0] if closest_poses[1] is None else closest_poses[1]
            pose_stamp = pose_msg.header.stamp

            if 'b' in self.states:
                ## find brightness closest to pose time stamp
                closest_brightnesses = [self.brightness_cache.getElemBeforeTime(pose_stamp),
                self.brightness_cache.getElemAfterTime(pose_stamp)]
                brightness_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_brightnesses]
                brightness = closest_brightnesses[np.argmin(brightness_times)].brightness
            else:
                brightness = None

            ## find joint closest to image time stamp
            # closest_joints = [self.joint_cache.getElemBeforeTime(pose_stamp),
            # self.joint_cache.getElemAfterTime(pose_stamp)]
            # joint_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_joints]
            # # print(pose_times)
            # joint_msg = closest_joints[np.argmin(joint_times)]
            # ee_state = joint_msg.position[self.ee_loc]

            ## find force message closest to image time stamp
            closest_forces = [self.force_cache.getElemBeforeTime(pose_stamp),
            self.force_cache.getElemAfterTime(pose_stamp)]
            force_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_forces]
            # print(pose_times)
            force_msg = closest_forces[np.argmin(force_times)]

            ## find vel message closest to image time stamp
            closest_vels = [self.vel_cache.getElemBeforeTime(pose_stamp),
            self.vel_cache.getElemAfterTime(pose_stamp)]
            vel_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_vels]
            # print(pose_times)
            vel_msg = closest_vels[np.argmin(vel_times)]

        ## process messages
        force = np.linalg.norm(from_vector3(force_msg.wrench.force),keepdims=True)
        ee_pose,full_ee_pos = self.process_pose_msg(pose_msg.pose,vel_msg.twist,brightness)
        return [ee_pose, full_ee_pos, force]

    def fix_yaw(self,val=0.2):
        cmd = Twist()
        cmd.angular.z = val
        send_cmd = rospy.ServiceProxy('/klerg_cmd', UpdateVel)
        self.reset_pub.publish()
        pose_msg = send_cmd(cmd,-1) # move vel
        time.sleep(self.dt*2)


    def check_goal_pos(self,tray_pos,brightness):
        # check joints 
        if self.pybullet:
            self.franka_env.resetJointsCallback(None)
        else:
            joint_msg = rospy.wait_for_message('/joint_states',JointState)
            if np.sum(joint_msg.position[:2]) < -2.: 
                self.reset_pub.publish()
                self.reset_joints_pub.publish()
                print('resetting joints')
                time.sleep(3)
        # then run loop
        at_center = False
        num_tries = 100
        attempt = 0
        tmp_pos = tray_pos.copy()
        tmp_cmd = self.format_Pose_msg(tmp_pos)
        pos_msg = self.send_cmd(tmp_cmd,brightness)
        while not rospy.is_shutdown() and (not at_center) and (attempt < num_tries):
            try:
                data,pos_check,full_pos,force,data_success = self.get_latest_msg()
                if not(pos_msg.success and data_success):
                    self.reset_pub.publish()
                    # print('reset')
                    data,pos_check,full_pos,force,data_success = self.get_latest_msg()
                # check if it's at the test location
                diff = pos_check-tray_pos
                at_center = np.all(abs(diff) < 0.02)
                if (not at_center) and (not self.pybullet):
                    skip_cmd = False
                    tmp_diff = np.abs(diff.copy())
                    sum_xyz_diff = np.sum([d for d,s in zip(tmp_diff,self.states) if s in 'xyz'])
                    sum_rpw_diff = np.sum([d for d,s in zip(tmp_diff,self.states) if s in 'rpw'])
                    tmp_pos = tray_pos.copy()
                    if ('w' in self.states):
                        w_idx = self.states.rfind('w')
                        if (abs(abs(pos_check[w_idx]) - np.pi/2) < 0.01):
                            # print('fix yaw',(abs(pos_check[w_idx]) - np.pi/2))
                            self.reset_ctrls_pub.publish() # clear pose command
                            self.fix_yaw(val=-0.5*np.sign(diff[w_idx]))
                            self.reset_ctrls_pub.publish() # clear vel command
                            skip_cmd = True
                    if not skip_cmd:
                        if (sum_xyz_diff > 0.2): # distance is too far for single step
                            # print('change pose')
                            for idx,s in enumerate(self.states):
                                if s in 'xyz':
                                    tmp_pos[idx] = pos_check[idx] - np.clip(diff[idx],-0.1,0.1)
                        if (sum_rpw_diff > 1.): # distance is too far for single step
                            # print('change angle')
                            for idx,s in enumerate(self.states):
                                if s in 'rpw':
                                    tmp_pos[idx] = pos_check[idx] - np.clip(diff[idx],-1.0,1.0)
                        tmp_cmd = self.format_Pose_msg(tmp_pos)
                        pos_msg = self.send_cmd(tmp_cmd,brightness)
                    attempt += 1
                    if not self.pybullet:
                        time.sleep(self.dt)
                    # print(attempt,diff,sum_xyz_diff,sum_rpw_diff,pos_check[w_idx],tray_pos[w_idx])
                if (not at_center) and (self.pybullet):
                    pos_msg = self.send_cmd(tmp_cmd,brightness)
                    attempt += 1
            except rospy.ServiceException as e:
                self.pause_pub.publish()
                if not self.pybullet:
                    time.sleep(0.1)
        return at_center


    def check_cmd(self,pos): 
        success = True
        ### check if the robot moved
        if (self.use_vel) and (self.use_force) and (self.last_state is not None) and (np.linalg.norm(self.last_state-pos) < 1e-5): 
            print('stuck',pos)
            success = False
            cmd = self.vel_move_force_norm()
            pos_msg = self.send_cmd(cmd,-1)
        elif (self.last_state is not None) and (np.linalg.norm(self.last_state-pos) < 1e-5):
            # print('stuck',pos)
            # success = False
            self.reset_pub.publish()
        self.last_state = pos
        return success


    def vel_move_force_norm(self): 
        force_msg = self.force_cache.getElemAfterTime(self.force_cache.getLatestTime())
        force = np.array([force_msg.wrench.force.x,force_msg.wrench.force.y,force_msg.wrench.force.z])
        # force /= self.max_force
        # force /= np.linalg.norm(force)
        force *= np.max(self.tray_ctrl_lim[:,1])*0.1
        
        for idx, state in enumerate('xyz'):
            if state in self.states: 
                s_idx = self.states.rfind(state)
                force[idx] = np.clip(force[idx],*self.tray_ctrl_lim[s_idx]*0.5)
            else: 
                force[idx] = 0.

        cmd = Twist()
        cmd.linear = Point(*force)
        return cmd


    def get_latest_msg(self,make_square=True):
        if self.use_pybullet: 
            pose_msg,vel_msg,force_msg,brightness,tmp_cam_img=self.franka_env.get_state()
            success = True
        else:
            tmp_cam_img = copy.copy(self.cam_img)
            success = (tmp_cam_img is not None)
            img_stamp = tmp_cam_img.header.stamp
            now = rospy.get_rostime()
            if (now.secs-img_stamp.secs) > 1: # check for lost connections
                self.got_img = False
                success = False
                print('image timed out')

            ## find pose closest to image time stamp
            closest_poses = [self.pose_cache.getElemBeforeTime(img_stamp),
            self.pose_cache.getElemAfterTime(img_stamp)]
            pose_times = [(np.abs(img_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_poses]
            pose_msg = closest_poses[np.argmin(pose_times)]
            if (pose_msg == None): # out of sync so just use latest pose
                pose_msg = closest_poses[-1]
                success = False
            pose_stamp = pose_msg.header.stamp

            if 'b' in self.states:
                ## find brightness closest to image time stamp
                closest_brightnesses = [self.brightness_cache.getElemBeforeTime(img_stamp),
                self.brightness_cache.getElemAfterTime(img_stamp)]
                brightness_times = [(np.abs(img_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_brightnesses]
                brightness = closest_brightnesses[np.argmin(brightness_times)].brightness
            else:
                brightness = None

            ## find joint closest to image time stamp
            # closest_joints = [self.joint_cache.getElemBeforeTime(pose_stamp),
            # self.joint_cache.getElemAfterTime(pose_stamp)]
            # joint_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_joints]
            # # print(pose_times)
            # joint_msg = closest_joints[np.argmin(joint_times)]
            # ee_state = joint_msg.position[self.ee_loc]

            ## find force message closest to image time stamp
            closest_forces = [self.force_cache.getElemBeforeTime(pose_stamp),
            self.force_cache.getElemAfterTime(pose_stamp)]
            force_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_forces]
            # print(pose_times)
            force_msg = closest_forces[np.argmin(force_times)]
            if (force_msg == None): # out of sync so just use latest force
                force_msg = closest_forces[-1]
                success = False

            ## find vel message closest to image time stamp
            closest_vels = [self.vel_cache.getElemBeforeTime(pose_stamp),
            self.vel_cache.getElemAfterTime(pose_stamp)]
            vel_times = [(np.abs(pose_stamp - p.header.stamp) if p is not None else rospy.Duration(10.)) for p in closest_vels]
            # print(pose_times)
            vel_msg = closest_vels[np.argmin(vel_times)]
            if (vel_msg == None): # out of sync so just use latest force
                vel_msg = closest_vels[-1]
                success = False

        ## process messages
        force = np.linalg.norm(from_vector3(force_msg.wrench.force),keepdims=True)
        if success:
            cam_img = self.process_image_msg(tmp_cam_img,make_square)
        else:
            cam_img = None
        ee_pose,full_ee_pos = self.process_pose_msg(pose_msg.pose,vel_msg.twist,brightness)
        return [cam_img, ee_pose, full_ee_pos, force, success]

        # return [self.cam_img.copy(), self.ee_pose.copy()]

    def image_callback(self,image_msg):
        rospy.loginfo_once("got image")
        self.cam_img = image_msg
        self.got_img = True

    def manual_callback(self,msg):
        rospy.logwarn("got manual message")
        self.manual = True

    def disable_manual_callback(self,msg):
        rospy.logwarn("got disable manual message")
        self.manual = False

    def pause_callback(self,msg):
        # rospy.logwarn("got pause message")
        self.pause = True
        if hasattr(self,'vae_buffer'):
            self.vae_buffer.pause()

    def resume_callback(self,msg):
        # rospy.logwarn("got resume message")
        self.pause = False
        if hasattr(self,'vae_buffer'):
            self.vae_buffer.resume()

    def save_callback(self,msg):
        rospy.logwarn("got save message")
        self.save(callback=True)

    def step(self,*args, **kw):
        pass

    def updateMarker(self,pos,rot):
        # define sensor dirction
        upVec = np.array([0,0,1])
        r  = Rotation.from_euler('xyz',rot).as_matrix()
        camDir = r @ upVec # get direction of sensor z-axis
        scale = pos[-1] / (camDir[-1] + 1e-5) # get z scaling to height of sensor

        xyz = pos - scale * camDir

        msg_pose = Pose(Point(*xyz),Quaternion(0,0,0,1.))

        robotMarker = Marker(header=Header(0,rospy.get_rostime(),"panda_link0"),
                            ns="robot",id=self.marker_count,type=2,action=0,
                            pose=msg_pose,scale=Point(0.005,0.005,0.01),
                            color=ColorRGBA(0,0,1.,1.),lifetime=rospy.Duration(100))

        self.marker_pub.publish(robotMarker)
        self.marker_count+=1

    def resetHistoryCallback(self,event):
        if self.robot is not None:
            self.robot.memory_buffer.reset()
        if self.use_pybullet:
            self.franka_env.moveObjectsCallback(None)
