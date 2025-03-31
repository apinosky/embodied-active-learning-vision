#!/usr/bin/env python

import torch
import numpy as np
import rospy

import datetime
import time

import os
from argparse import Namespace
import yaml

from termcolor import cprint, colored
from dist_modules.utils import add_yaml_representers

add_yaml_representers()

def get_config(print_output=True):
    args = Namespace()

    # modes
    explr_methods = ['entklerg', 'uniform', 'randomWalk','unifklerg']
    sensor_methods = ['rgb', 'intensity']
    args.dtype = 'torch.float32'  # note float32 precision for cpu and gpu are different so be careful about mixing these
    ddp = rospy.get_param('ddp', False)
    args.distributed = rospy.get_param('distributed', False)
    args.other_locs = rospy.get_param('other_locs', True) # if trainer should pass different locations to the decoder other than the "seed" location (True is slowerer than False)
    args.async_trainer =  rospy.get_param('async', False)
    if args.async_trainer:
        # assert ddp, f"must use ddp_trainer for async mode"
        args.ddp_model = False
    else:
        args.ddp_model = ddp
    args.ddp_trainer = ddp

    # params
    args.num_update_proc = rospy.get_param('num_trainers', '2') if ddp else 1
    args.explr_method = rospy.get_param('explr_method', 'entklerg')
    args.states = rospy.get_param('explr_states', 'xyw')
    args.sensor_method = rospy.get_param('sensor_method', 'intensity')
    args.sensor_mod = rospy.get_param('sensor_mod', '')
    args.seed = rospy.get_param('seed', 0)
    args.pybullet = rospy.get_param('pybullet', False)
    args.dt = rospy.get_param('dt', 0.1)
    args.cam_z = rospy.get_param('cam_z', 'N/A')
    args.path_mod = rospy.get_param('path_mod', '')
    args.test_config_file = rospy.get_param('test_config_file', 'test_config.yaml')

    # check modes
    assert args.explr_method in explr_methods, f"requested invalid exploration method '{args.explr_method}', choose from {explr_methods}"
    assert args.sensor_method in sensor_methods, f"requested invalid sensor method '{args.sensor_method}', choose from {sensor_methods}"
    args.intensity = (args.sensor_method == 'intensity')
    args.learn_force = False
    args.use_force = False 

    # set up test config
    # load robot config and robot config
    base_path = rospy.get_param('base_path', './')
    with open(base_path+'/config/' + args.test_config_file) as f:
        tmp = yaml.load(f,Loader=yaml.FullLoader)
    for _, param in tmp.items(): # top level is for user readability
        for k, v in param.items():
            setattr(args, k, v)

    if args.print_debug and print_output: cprint(f'*** Pybullet = {args.pybullet} ***','yellow')

    # fix param
    args.frames_before_training = max(args.batch_size,args.frames_before_training)
    # if 'erg' not in args.explr_method: args.horizon = 1

    # cuda
    if torch.cuda.is_available() and args.use_gpu:
        args.device = 'cuda:0'
        if print_output: cprint('Using GPU','green')
        if torch.cuda.device_count() < args.num_update_proc:
            if print_output: cprint(f'Requested more update processes than available cuda devices, so changing from {args.num_update_proc} to {torch.cuda.device_count()}','red')
            args.num_update_proc = torch.cuda.device_count()
    else:
        args.device = 'cpu'
        args.use_gpu = False

    if args.num_update_proc == 1:
        ddp = False
        args.ddp_model = False
        args.ddp_trainer = False

    # check states 
    assert len(args.plot_states)==2, "invalid number of plot states requested (need 2 states)"
    assert all([p in args.states for p in args.plot_states]), "requested plot_state not included in states"
    assert all([args.states.count(c) == 1 for c in args.plot_states]), "duplicate plot_states detected, check requested states"
    assert all([args.states.count(c) == 1 for c in args.states]), "duplicate state detected, check requested states"
    assert all([p.lower() in args.states for p in args.states if p == p.upper()]), "invalid velocity requested (in states). velocities require positions (e.g. xY is not valid but xX is)"

    # rospy.set_param('explr_states', args.states) 

    # first filter the states so you can keep xyzrpw all the test_config file
    lower_states        = [s for s in args.states if s == s.lower()]
    state_subset_locs   = [args.raw_states.rfind(s) for s in lower_states]
    args.tray_lim       = [args.tray_lim[s] for s in state_subset_locs]
    args.robot_lim      = [args.robot_lim[s] for s in state_subset_locs]
    args.tray_ctrl_lim  = [args.tray_ctrl_lim[s] for s in state_subset_locs]
    args.robot_ctrl_lim = [args.robot_ctrl_lim[s] for s in state_subset_locs]

    assert isinstance(args.R,float) or len(args.robot_lim) == len(args.R), 'number of items in R must match robot_lim or be a single float'
    args.vel_states = False
    if not (args.states == args.states.lower()): # check for capital letters
        args.vel_states = True
        for s in args.states:
            if not (s == s.lower()):
                idx = args.states.rfind(s.lower())
                args.tray_lim = args.tray_lim + [args.tray_ctrl_lim[idx]]
                args.robot_lim = args.robot_lim + [args.robot_ctrl_lim[idx]]
    args.s_dim = len(args.states)
    assert len(args.tray_lim) == args.s_dim, 'tray_lim must match number of states'
    assert len(args.robot_lim) == args.s_dim, 'robot_lim must match number of states'
    # assert args.s_dim % len(args.tray_ctrl_lim) == 0, 'tray_ctrl_lim must match number of states'
    # assert args.s_dim % len(args.robot_ctrl_lim) == 0, 'robot_ctrl_lim must match number of states'
    # assert len(args.tray_ctrl_lim) == args.s_dim, 'tray_ctrl_lim must match number of states'
    # assert len(args.robot_ctrl_lim) == args.s_dim, 'robot_ctrl_lim must match number of states'
    # if 'p' in args.states:
    #     assert ('r' in args.states and 'w' in args.states), 'cannot use pitch without roll and yaw'
    # if 'r' in args.states:
    #     assert ('p' in args.states and 'w' in args.states), 'cannot use roll without pitch and yaw'

    if args.dx and np.sum([1 for s in args.states if s in 'rpw']) > 1: 
        raise NotImplemented('dx for rotations with real angles (i.e. more than one angle) not implemented yet')

    # get std from ratio (scales std based on limits of space, number of dimensions, and num_target_samples)
    from scipy.special import gamma
    def get_std(lims,desired_ratio): 
        n = lims.shape[0]
        vol_n_cube = (lims[:,1] - lims[:,0]).prod()
        std = ( desired_ratio * vol_n_cube * gamma(n/2+1) / np.pi**(n/2) ) ** ( 1/n )
        return std.item()
    desired_ratio = 0.1/args.num_target_samples
    lims = np.array(args.robot_lim)
    args.std = get_std(lims,desired_ratio) # previously was 0.05
    args.plot_idx = [args.states.rfind(s) for s in args.plot_states]
    # args.std_plot = get_std(lims[args.plot_idx],desired_ratio) # previously was 0.05
    args.std_plot = args.std

    # args.std = 0.05
    # args.std_plot = 0.05

    # set up test config
    args.raw_image_dim = args.image_dim.copy()
    args.image_dim = np.flip(args.image_dim)
    args.down_sample = max(args.down_sample,1)
    args.zoom = max(args.zoom,1)
    def update_dim(image_dim,down_sample,zoom):
        image_dim[1:] = image_dim[1:]//down_sample #+1
        image_dim[1:] = image_dim[1:]//zoom #+1

    from vae.vae import get_input_dim
    update_dim(args.image_dim,args.down_sample,args.zoom)
    if args.intensity:
        args.image_dim[0] = 1.
    if args.flat_sensor: 
        args.CNNdict = None
    input_dim_prod,args.input_dim = get_input_dim(args.image_dim,args.CNNdict) # (for reference no icra used)
    ## check if network needs more FC layers
    done = False 
    max_scale_per_layer = 8
    while not done: 
        if input_dim_prod / args.hidden_dim[0] > max_scale_per_layer: 
            scale = int(min(np.ceil(np.sqrt(input_dim_prod / args.hidden_dim[0])),max_scale_per_layer))
            args.hidden_dim = [args.hidden_dim[0]*scale] + args.hidden_dim
        else: 
            done = True
    args.image_dim = args.image_dim.tolist()
    args.input_dim = args.input_dim.tolist()
    # if  args.print_debug and print_output: 
    cprint([input_dim_prod,args.hidden_dim,args.input_dim,args.image_dim],'blue')

    # Set up vars for saving
    if args.pybullet:
        base_path += "/sim_data/"
    else:
        base_path += "/data/"

    dir_path = base_path + "{}{}/{}_{:04d}{}/".format(args.sensor_method,args.sensor_mod,args.explr_method,args.seed,args.path_mod)

    if print_output:
        if os.path.exists(dir_path) == False:
            os.makedirs(dir_path)
        if len(os.listdir(dir_path)) > 1: # if clusters goes first there will be 1 item
            check_path = input(colored('this path already exists, are you sure you want to proceed?  Y/N (N will quit)\n','red'))
            if ('n' in check_path.lower()):
                cprint('got N ... aborting ...','red')
                raise Exception('aborting')

        # write config to file
        now = datetime.datetime.now()
        date_str = now.strftime("%Y-%m-%d_%H-%M-%S/")
        with open(dir_path + "log.txt","a") as f:
            f.write(date_str+'\n')
        if args.print_debug and print_output: cprint(vars(args),'blue')
        with open(dir_path + "config.yaml","w") as f:
            yaml.dump(args.__dict__,f)

    # convert from string after saving config
    args.dir_path = dir_path
    args.dtype = eval(args.dtype)
    args.tray_lim = np.array(args.tray_lim)
    args.robot_lim = np.array(args.robot_lim)
    args.tray_ctrl_lim = np.array(args.tray_ctrl_lim)
    args.robot_ctrl_lim = np.array(args.robot_ctrl_lim)

    return args
