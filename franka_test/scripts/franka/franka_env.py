#!/usr/bin/env python

import numpy as np
import math

import pybullet as p
import pybullet_data as pd
import os
import time
from tf_conversions import transformations

#restposes for null space
# jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30., 2.66, 2.32, 0.02, 0.02] # orig
# jointPositions=[0., 0.458, 0.31, -2.24, -0.30., 2.66, 2.32, 0.02, 0.02] # rotated to match physical setup
# jointPositions=[0., 0., 0., -2.24, -0.30, 2.66, 2.32] # rotated to match physical setup
jointPositions=[0.076, -0.17, -0.07, -2.16, -0.0133, 1.99, -0.38]


class FrankaEnv(object):
    def __init__(self, render=True, ts=1./60., offset=[0.,0.,0.075],
                 x0 = [0.,0.,0.],speckle=False,img_shape=[240,320],
                 xlim=[0.325,0.625],ylim=[-0.15,0.15],camera_view=None,
                 show_ws_lines=True,sensor_method = 'rgb',
                 soft_objects=False,base_path=''): 

        self.render = render
        self.cam_sensors =  ['rgb','camera','cam','intensity']
        assert sensor_method in self.cam_sensors,f'invalid sensor requested'
        self.img_shape = img_shape
        self.xlim = xlim
        self.ylim = ylim
        self.bullet_client = p
        self.base_path = base_path + '/urdf/'
        self.offset = np.array(offset)
        self.simulation_rate = 1/60.
        self.dt = ts
        self.iters_per_step = max(1,int(ts/self.simulation_rate))
        self.soft_objects = soft_objects

        self.build_scene(x0,speckle,show_ws_lines,camera_view=camera_view)

        self.update_ee_state()

        # print(self.curr_pos, self.curr_orn)

    def build_scene(self,x0,speckle,show_ws_lines,test_cubes=False,camera_view=None): 
        # Set up Pybullet Environment
        if self.render:
            self.bullet_client.connect(self.bullet_client.GUI,options='--width={} --height={}'.format(500,500))
        else:
            self.bullet_client.connect(self.bullet_client.DIRECT)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_Y_AXIS_UP,0)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_GUI, 0)
        self.bullet_client.configureDebugVisualizer(self.bullet_client.COV_ENABLE_SHADOWS,0) # added
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.bullet_client.setTimeStep(self.simulation_rate) # match speed of real franka updates
        # self.bullet_client.setGravity(0,0,-0.01) # essentially ignore gravity
        self.bullet_client.setGravity(0,0,-9.8)
        self.bullet_client.setPhysicsEngineParameter()#allowedCcdPenetration=0.0)

        if self.render: 
            camera_views = {'top':[0.6,0,-89.99,[0.4,-0.,0.2]], # dist, pitch, yaw, camTargetPos
                            'tilt':[0.7,15.,-30,[0.3,-0.,0.2]], 
                            'side_Rleft': [0.4,0,-5,[0.4,-0.3,0.2]],
                            'side_Rright':[0.7,180,-30,[0.4,0.3,0.2]],
                            'forward':[1.3,450.4,-40.60,[0,0,-0.45]],
                            'behind':[1.0,-90,-30,[0.3,0.2,0.]]}
            # move GUI camera and add axes
            if camera_view is None: 
                camera_view='tilt'
            self.bullet_client.resetDebugVisualizerCamera(*camera_views[camera_view]) 

            self.bullet_client.addUserDebugLine([0.25,0,0.01],[0,0,0.01],[0.5,0,0],lineWidth=4) # red x
            self.bullet_client.addUserDebugText('x',[0.25,0,0.01],[0.5,0,0]) # red x
            self.bullet_client.addUserDebugLine([0,0.25,0.01],[0,0,0.01],[0.2,0.5,0.2],lineWidth=4) # green y
            self.bullet_client.addUserDebugText('y',[-0.02,0.25,0.01],[0.2,0.5,0.2]) # green y

            off = 0.1
            off_lim = np.array([-off,+off])
            xlim = self.xlim.copy() + off_lim
            ylim = self.ylim.copy() + off_lim
            if not(all(self.offset==0)): 
                halfextents = [(xlim[1]-xlim[0])/2,(ylim[1]-ylim[0])/2,self.offset[2]/2]
                visualShapeId = self.bullet_client.createVisualShape(shapeType=self.bullet_client.GEOM_BOX,rgbaColor=[1, 1, 1, 1],halfExtents=halfextents)
                collisionShapeId = self.bullet_client.createCollisionShape(shapeType=self.bullet_client.GEOM_BOX,halfExtents=halfextents)
                offsetID = self.bullet_client.createMultiBody(0.,baseInertialFramePosition=[0, 0, 0],
                                baseVisualShapeIndex=visualShapeId,
                                baseCollisionShapeIndex=collisionShapeId,
                                basePosition=[xlim[0] + halfextents[0],ylim[0] + halfextents[1], halfextents[2]],
                                useMaximalCoordinates=True)
            elif show_ws_lines:
                self.bullet_client.addUserDebugLine([xlim[1],ylim[1],0.01],[xlim[1],ylim[0],0.01],[0,0,0],lineWidth=4) # black boundary
                self.bullet_client.addUserDebugLine([xlim[1],ylim[0],0.01],[xlim[0],ylim[0],0.01],[0,0,0],lineWidth=4) # black boundary
                self.bullet_client.addUserDebugLine([xlim[0],ylim[0],0.01],[xlim[0],ylim[1],0.01],[0,0,0],lineWidth=4) # black boundary
                self.bullet_client.addUserDebugLine([xlim[0],ylim[1],0.01],[xlim[1],ylim[1],0.01],[0,0,0],lineWidth=4) # black boundary

        # Set up Franka
        orn=self.bullet_client.getQuaternionFromEuler([0,0,0]) # rotated to match physical setup
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        file_name = 'panda_camera.urdf'
        self.robot = self.bullet_client.loadURDF(self.base_path+file_name, np.array(x0), orn, useFixedBase=True, flags=flags) 
        self.numJoints = self.bullet_client.getNumJoints(self.robot)
        for x in range(self.numJoints):
            self.bullet_client.changeVisualShape(self.robot, linkIndex=x, rgbaColor= [1.,1.,1.,0.75] if x < self.numJoints-1 else [0.,0.,0.,0.8] ) # make robot transparent
        self.ee_ind = self.numJoints-1 # was fixed at joint 7 before adding sensors
        self.robot_dof = 7
        self.bullet_client.enableJointForceTorqueSensor(self.robot, self.ee_ind-1, True) # get force for ee

        for j in range(self.numJoints):
            self.bullet_client.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)
        self.reset()

        self.brightness = 1.0
        # Set Camera Properties
        self.projMat = self.bullet_client.computeProjectionMatrixFOV(fov=30.0, aspect=float(self.img_shape[0])/float(self.img_shape[1]), nearVal=0.01, farVal=100.)


        # Set up table/objects in Environment
        table_orn=self.bullet_client.getQuaternionFromEuler([0.,0.,0.]) # rotated
        table_pos = [0.4, 0., -0.625]
        self.tableID = self.bullet_client.loadURDF(self.base_path+"table_plain.urdf", np.array(table_pos),table_orn, flags=flags)
        # self.bullet_client.changeVisualShape(self.tableID, linkIndex=-1, rgbaColor=[0.87, .721, .529, 1]) # brown table
        if speckle:
            x = self.bullet_client.loadTexture(self.base_path+"tex.png")
            self.bullet_client.changeVisualShape(self.tableID, linkIndex=-1, rgbaColor=[0.95,0.95,0.95,1.], textureUniqueId=x) # textured table
        else:
            x = self.bullet_client.loadTexture(self.base_path+"tex2.jpg")
            self.bullet_client.changeVisualShape(self.tableID, linkIndex=-1, rgbaColor=[0.95,0.95,0.95,1.], textureUniqueId=x) # textured table
            # self.bullet_client.changeVisualShape(self.tableID, linkIndex=-1, rgbaColor=[1.,1.,1.,1.]) # textured table

        self.objects = ['duck','cube','sphere','pineapple']
        self.objID = []
        self.objPATH = []
        
        # "duck"
        scene_objs = ["duck"]
        poses = [[0.38,-0.1, 0.03]]
        orns = [self.bullet_client.getQuaternionFromEuler([np.pi,0.,math.pi/3])] 
        colors = [[0.88,0.29,0.376,1.0]]

        # cubes
        if test_cubes:
            poses = poses + [[0.48,0.13,0.05],[0.35,0.,0.05]]
            colors = colors + [[-0.03321029, -1.1062515 , -0.64966736, 1. ],[ 0.4406284 ,  1.81443231, -0.09169206,  1.]]
            orns = orns + [[ 0.10854133, -0.02847903,  0.66694362,  0.25299408],[ 4.32799769, -3.62206671,  1.59637661, -1.30230759]]
            scene_objs = scene_objs + ["cube","cube"]

        # "plant"
        scene_objs.append("pineapple") # changed plant to this template because kept tipping over
        poses.append([0.52,0.08, 0.03])
        orns.append(self.bullet_client.getQuaternionFromEuler([0.,0.,0.]))
        colors.append([0.,0.3,0.,1.])

        for obj_name,obj_pose,obj_orn,color in zip(scene_objs,poses,orns,colors):
            self.load_object(obj_name,np.array(obj_pose),obj_orn,color)

    def load_object(self,obj_name,obj_pose,obj_orn,color=None,useFixedBase=True): 
        urdf_path = self.base_path+obj_name+'.urdf'
        if self.soft_objects:
            scale = 1.2 if obj_name == 'pineapple' else 0.05
            objID = self.bullet_client.loadSoftBody(self.base_path+'/meshes/'+obj_name+'.obj', basePosition = obj_pose+self.offset, baseOrientation=obj_orn, scale = scale, mass = 2.25, useNeoHookean = 1, NeoHookeanMu = 400, NeoHookeanLambda = 600, NeoHookeanDamping = 0.001, useSelfCollision = 1, frictionCoeff = .5, collisionMargin = 0.001)
        else:
            scale = 0.9
            objID = self.bullet_client.loadURDF(urdf_path,obj_pose+self.offset,obj_orn,useFixedBase=useFixedBase,useMaximalCoordinates=True,globalScaling = scale)
        if color is not None: 
            self.bullet_client.changeVisualShape(objID, linkIndex=-1, rgbaColor=color)
        self.objID.append(objID)
        self.objPATH.append(urdf_path)

        return objID

    def move_objects(self):
        for objID in self.objID:
            pos,orn = self.bullet_client.getBasePositionAndOrientation(objID)
            pos = np.array(pos)
            pos[0] = np.random.uniform(*self.xlim)
            pos[1] = np.random.uniform(*self.ylim)
            self.bullet_client.resetBasePositionAndOrientation(objID,pos+self.offset,orn)
            
    def update_brightness(self,brightness):
        self.brightness = brightness

    def add_object(self):
        obj = np.random.choice(self.objects)
        pos = np.zeros(3)
        pos[0] = np.random.uniform(*self.xlim)
        pos[1] = np.random.uniform(*self.ylim)
        orn = np.random.randn(4); orn/=np.sum(orn) # random

        color = np.hstack([np.random.randn(3),np.ones(1)]) # random
        self.load_object(obj,pos,orn,color)

    def reset(self):
        index = 0
        for j in range(self.numJoints):
            info = self.bullet_client.getJointInfo(self.robot, j)
            jointName = info[1]
            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.robot, j, jointPositions[index])
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.robot, j, jointPositions[index])
                index=index+1
    
    def step(self, pos, orn, leave_trace = False, use_vel=False,save_update=True):
        num_iters = self.iters_per_step if save_update else 1
        for _ in range(num_iters):
            if use_vel: ## note: this sometimes drifts, so you may need to level the end effector occasionally if you're exploring in the plane
                ee_velocity = np.hstack([pos,orn])
                # get current joints
                joint_pos = [state[0] for state in self.bullet_client.getJointStates(self.robot, range(self.robot_dof))]
                # get jacobian
                eeState = self.bullet_client.getLinkState(self.robot, self.ee_ind, computeForwardKinematics=True)
                link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot = eeState
                zero_vec = [0.0] * len(joint_pos)
                jac_t, jac_r = self.bullet_client.calculateJacobian(self.robot, self.ee_ind, com_trn, list(joint_pos), zero_vec, zero_vec)
                J = np.concatenate((np.asarray(jac_t), np.asarray(jac_r)), axis=0)
                # get joint velocities
                joint_velocities = np.linalg.pinv(J) @ ee_velocity
                # update simulator
                self.bullet_client.setJointMotorControlArray(self.robot,
                                range(self.robot_dof),
                                self.bullet_client.VELOCITY_CONTROL,
                                targetVelocities=joint_velocities,)
                                # forces = [0] * (self.ee_ind))
            else:
                jointPoses = self.bullet_client.calculateInverseKinematics(self.robot,self.ee_ind, pos, orn, maxNumIterations=50,jointDamping = [0.1]*(self.ee_ind-1)+[0.05],residualThreshold=0.01)
                # update simulator
                self.bullet_client.setJointMotorControlArray(self.robot,
                                range(len(jointPoses)),
                                self.bullet_client.POSITION_CONTROL,
                                targetPositions=jointPoses,
                                targetVelocities=[0.0]*len(jointPoses), )
                                # forces = [500] * len(jointPoses))
            self.bullet_client.stepSimulation()
            if save_update:
                self.update_ee_state()

        if leave_trace:
            # self.bullet_client.loadURDF(self.base_path + "/urdf/sphere.urdf",eePos, eeOrn,useFixedBase=True,globalScaling=0.2)
            self.bullet_client.addUserDebugText("+", [self.curr_pos[0], self.curr_pos[1], 0.],textColorRGB=[0, 0, 0], textSize=2, lifeTime=10)
        pass

    @property
    def current_brightness(self): 
        return self.brightness

    @property
    def cam_img(self):
        _camera_offset = [0.,0.,0.04] # camera is actually lower than last joint
        pos,orn = self.bullet_client.getLinkState(self.robot, self.numJoints-1)[:2]
        T = transformations.translation_matrix(pos) @ transformations.quaternion_matrix(orn) @ transformations.translation_matrix(_camera_offset)
        # convert from OpenCV coordinate frame to OpenGL coordinate frame
        # rotate 180 deg about x-axis (have y and z point in the opposite direction)
        T[:,1:3] *= -1
        T = np.linalg.inv(T)
        # serialise column-wise
        viewMatrix = T.T.ravel()

        light_args = {'shadow':False,'lightDistance':2.5 ,'lightDirection':[0.5,0.,0.8],'lightAmbientCoeff':self.brightness*0.6,'lightColor':(1.0,1.0,1.0),'renderer':p.ER_TINY_RENDERER} # defaults are ... 'lightSpecularCoeff':0.05,'lightDiffuseCoeff':0.35, 'lightAmbientCoeff':0.6}, renderers = {p.ER_BULLET_HARDWARE_OPENGL,p.ER_TINY_RENDERER}
        width, height, rgbImg, depthImg, segImg = self.bullet_client.getCameraImage(width=self.img_shape[0], height=self.img_shape[1], viewMatrix=viewMatrix,projectionMatrix=self.projMat,**light_args)
        # rgbImg = np.flipud(rgbImg) # if webcam flipped in real world
        rgbImg = np.fliplr(rgbImg) # if webcam flipped in real world
        return rgbImg.copy()

    def update_ee_state(self):
        eeState = self.bullet_client.getLinkState(self.robot, self.ee_ind, computeForwardKinematics=True, computeLinkVelocity=True)
        eeJointState = self.bullet_client.getJointState(self.robot, self.ee_ind-1)
        eeOrn = np.array(eeState[1])
        eeForce = np.array(eeJointState[2])

        # gravity compensation                         
        gravity_force = np.array([0.,0.,-9.8*0.2])
        R = transformations.quaternion_matrix(eeOrn)[:3,:3]
        gravity_force = R.T @ gravity_force
        eeForce[:3] += gravity_force

        self.curr_pos = np.array(eeState[0])
        self.curr_orn = eeOrn
        self.curr_lin_vel = np.array(eeState[6])
        self.curr_ang_vel = np.array(eeState[7])
        self.curr_force = eeForce

    
    def get_joint_states(self):
        return self.bullet_client.getJointStates(self.robot,range(self.robot_dof))

    def __del__(self):
        self.bullet_client.disconnect()

if __name__ == "__main__":

    sensor_method  = "rgb" 
    base_path = os.path.dirname(os.path.realpath(__file__))

    # # Initialize Franka Robots
    timeStep=1./100.
    offset = [0.,0.,0.]
    cam_z = 0.415
    dz = -1.
        
    env = FrankaEnv(render=True, ts=timeStep,offset=offset,img_shape=[180,180],sensor_method=sensor_method,base_path=base_path)
    start_x, start_y =  [0.42,-0.15]  #  [0.55,-0.1] #
    pos = np.array([start_x, start_y, cam_z])
    rpw = np.array([math.pi,0,-math.pi/8.])
    orn = env.bullet_client.getQuaternionFromEuler(rpw)
    for _ in range(20):
        env.step(pos, orn, use_vel=False)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.show(block=False)
    # brightness command
    for i in range(20):
        brightness = 1.-i/20.
        env.update_brightness(brightness)
        env.step(np.zeros(3),np.zeros(3), use_vel = True, leave_trace = False)
        plt.imshow(env.cam_img)
        plt.title(brightness)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    for i in range(20):
        brightness = i/20.
        env.update_brightness(brightness)
        env.step(np.zeros(3),np.zeros(3), use_vel = True, leave_trace = False)
        plt.imshow(env.cam_img)
        plt.title(brightness)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    # vel command (rotate yaw)
    for i in range(20):
        env.step(np.zeros(3),np.array([0,0,10.0]), use_vel = True, leave_trace = True)
        plt.imshow(env.cam_img)
        rpw = np.array(env.bullet_client.getEulerFromQuaternion(env.curr_orn))
        plt.title('vel ctrl\npos: {:0.2f} {:0.2f} {:0.2f}\norientation: {:0.0f} {:0.0f} {:0.0f}'.format(*env.curr_pos,*(rpw*180/np.pi)))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # vel command (-z)
    for i in range(30):
        env.step(np.array([0,0,dz]),np.zeros(3), use_vel = True, leave_trace = True)
        plt.imshow(env.cam_img)
        rpw = np.array(env.bullet_client.getEulerFromQuaternion(env.curr_orn))
        plt.title('vel ctrl\npos: {:0.2f} {:0.2f} {:0.2f}\norientation: {:0.0f} {:0.0f} {:0.0f}'.format(*env.curr_pos,*(rpw*180/np.pi)))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    # vel command (+z)
    for i in range(15):
        env.step(np.array([0,0,-dz*2]),np.zeros(3), use_vel = True, leave_trace = True)
        plt.imshow(env.cam_img)
        rpw = np.array(env.bullet_client.getEulerFromQuaternion(env.curr_orn))
        plt.title('vel ctrl\npos: {:0.2f} {:0.2f} {:0.2f}\norientation: {:0.0f} {:0.0f} {:0.0f}'.format(*env.curr_pos,*(rpw*180/np.pi)))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # pose command (rotate yaw)
    pos = np.array(env.curr_pos)
    rpw = np.array(env.bullet_client.getEulerFromQuaternion(env.curr_orn))
    for i in range(20):
        rpw[0] = np.pi
        rpw[1] = 0.
        rpw[2] -= np.pi/20
        orn = env.bullet_client.getQuaternionFromEuler(rpw)
        env.step(pos,orn, use_vel = False,leave_trace = True)
        plt.imshow(env.cam_img)
        rpw = np.array(env.bullet_client.getEulerFromQuaternion(env.curr_orn))
        plt.title('pose ctrl\npos: {:0.2f} {:0.2f} {:0.2f}\norientation: {:0.0f} {:0.0f} {:0.0f}'.format(*env.curr_pos,*(rpw*180/np.pi)))
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # time.sleep(1.)
    # plt.show()
