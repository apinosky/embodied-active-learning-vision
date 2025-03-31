#!/usr/bin/env python

########## global imports ##########
import numpy as np
np.set_printoptions(precision=3)
import pickle
import yaml
import datetime
import time
import torch
from termcolor import cprint
from contextlib import suppress

import rospy

########## local imports ##########
from control.klerg_utils import *

from .utils import setup, build_arg_dicts, add_yaml_representers, get_num_cores
from .fingerprint_module import FingerprintID,FingerprintsPlotter
from .main_async import get_cost

add_yaml_representers()

video=False

class FingerprintBufferTorch(torch.nn.Module):
    __attributes__ = ['state_buffer','cam_data_buffer','iter_buffer']
    def __init__(self,capacity,x_dim,y_dim,dtype,device):
        super(FingerprintBufferTorch,self).__init__()

        self.capacity = capacity
        self.buffer_device = 'cpu'
        self.device = device
        self.dtype = dtype
        self.position = torch.tensor([0])
        self.full_buffer = torch.tensor([False])
        self.state_buffer = torch.empty(capacity,x_dim,dtype=self.dtype,device=self.buffer_device)
        self.cam_data_buffer = torch.empty(capacity,*y_dim,dtype=self.dtype,device=self.buffer_device)
        self.iter_buffer = torch.empty(capacity,dtype=torch.int64)

    def share_memory(self):
        self.position.share_memory_()
        self.full_buffer.share_memory_()
        self.state_buffer.share_memory_()
        self.cam_data_buffer.share_memory_()
        self.iter_buffer.share_memory_()

    def push(self, state, cam_data, force, count):
        if not(isinstance(cam_data,torch.Tensor)):
            cam_data = torch.as_tensor(cam_data).to(self.dtype)
        if not(isinstance(state,torch.Tensor)):
            state = torch.as_tensor(state).to(self.dtype)
        if (self.position + 1 ) == self.capacity:
            self.full_buffer[0] = True
        self.cam_data_buffer[self.position] = cam_data.to(self.buffer_device)
        self.state_buffer[self.position] = state.to(self.buffer_device)
        self.iter_buffer[self.position] = torch.as_tensor(count).to(self.buffer_device)
        self.position += 1
        self.position.remainder_(self.capacity)

    def __len__(self):
        return self.capacity if self.full_buffer else self.position

    def get_data(self,loc):
        position = torch.argwhere(self.iter_buffer==loc)[0].item()
        return [self.state_buffer[position].to(device=self.device),
                self.cam_data_buffer[position].to(device=self.device)]

    def extra_repr(self):
        return  '\n'.join(('{} = {}'.format(item, self.__dict__[item].shape if isinstance(self.__dict__[item],torch.Tensor) else self.__dict__[item]) for item in self.__attributes__))


def duration_str(start_time):
    return str(datetime.timedelta(seconds=(time.time()-start_time)))

def test_fingerprint(rank,world_size,seed,queue,fingerprint_buffer,test_config,in_tdist_queue,update_rate,num_steps):
    fp = FingerprintID(**test_config)

    # set up plotting
    name = fp.dist_method
    if fp.error:
        no_figs = 1
        fp_angle_method = ['max']
        end_str = ' (error)'
        end_save = '_error'
    else:
        no_figs = 2
        fp_angle_method = ['mean','max']
        end_str =  ""
        end_save = ""
    fig2 = []
    save_name = []
    fig_no = 0

    for x in test_config['fingerprint_path'].split('/'):
         if ('erg' in x) or ('unif' in x) or ('rand' in x):
             y = x.split('_')
             fp_base_name = ' '.join([y[0][:5] + str(int(y[1]))] + y[2:])

    # make figures
    base_window_name = fp_base_name +' '+name
    base_save_name = f"_heatmaps_{name}"
    for angle_method in fp_angle_method:
        fig2.append(FingerprintsPlotter(fp.fingerprint_names,fp.target_dists,fp.robot_lim[fp.plot_idx],
                        corner_samples=fp.corner_samples,
                        window_name= str(angle_method) + end_str + ' | '+ base_window_name ,
                        rank=rank,fig_no=fig_no,no_figs=no_figs,explr_dim=len(fp.plot_idx),render=fp.render_figs))
        save_name.append(base_save_name+f"_{angle_method}"+end_save)
        fig_no += 1

    if in_tdist_queue is not None:
        update_tdist_step = rospy.get_param("update_tdist_step", 200)
        explr_fingerprint = rospy.get_param("explr_fingerprint", 2)

    killer = setup(rank,world_size,seed,fp.use_gpu)
    print_name = f'[FINGERPRINT MODULE (error, RANK {rank})]' if fp.error else f'[FINGERPRINT MODULE (RANK {rank})]'
    cprint(f'{print_name} setup complete','magenta')
    done = False
    timeout = 0.1
    start = time.time()
    while not killer.kill_now and not done:
        try:
            if queue.poll(timeout=timeout):
                type,iter_step = queue.recv()
                if type=='update':
                    data = fingerprint_buffer.get_data(iter_step)
                    fp.test_fingerprints(*data,sync=iter_step % update_rate == 0) #state,img,update_debug_plots=(iter_step % 10 == 0))
                    if iter_step == update_rate:
                        for tdist in fp.target_dists:
                            tdist.init = True
                    if fig2 is not None and iter_step % update_rate == 0:
                        for idx,(fig,name,angle_method) in enumerate(zip(fig2,save_name,fp_angle_method)):
                            if fp.save_figs:
                                fname = fp.fpath+f"{name}_step{iter_step:05d}.png" # svg"
                            else:
                                fname = None
                            fig.update(fname,angle_method=angle_method,title=f'{iter_step}/{num_steps}')
                        if in_tdist_queue is not None and iter_step > update_tdist_step-1:
                            in_tdist_queue.send(fp.target_dists[explr_fingerprint])
                    elif fig2 is not None:
                        for fig in fig2:
                            fig.refresh()
                    if iter_step > 0 and iter_step % 250 == 0:
                        for tdist in fp.target_dists:
                            tdist.save_results(fp.fpath,f'step{iter_step:05d}')
                    if iter_step % update_rate == 0 :
                        cprint(f'{print_name} {iter_step}/{num_steps}  | {duration_str(start)}','magenta')
                elif type=='done':
                    done = True
                # del plot_data
        except BrokenPipeError:
             break
    for tdist in fp.target_dists :
        tdist.save_results(fp.fpath,'final')
    if fp.save_figs:
        if fig2 is not None:
            for idx,(fig,name,angle_method) in enumerate(zip(fig2,save_name,fp_angle_method)):
                if iter_step == None:
                    iter_step = num_steps
                fname = fp.fpath+f'{name}_step{iter_step:05d}.png' #.svg'
                fig.update(fname,angle_method=angle_method,title=f'{iter_step}/{num_steps}')
            with open(fp.fpath+'_config.yaml',"w") as f:
                yaml.dump(test_config,f)
    # close queue
    del queue
    if in_tdist_queue is not None:
        del in_tdist_queue
    cprint(f'{print_name} shutdown | {duration_str(start)}','magenta')

def test_main(rank,world_size,seed,queues,fingerprint_buffer,tdist_queue,plot_queue,plotter_buffer,explr_states,update_rate,args,replay_buffer,fpaths,num_threads_real):

    save_name = rospy.get_param('save_name', 'test')
    update_tdist_step = rospy.get_param("update_tdist_step", 200)
    new_model_explr = rospy.get_param("new_model_explr", False)
    use_async = rospy.get_param("async", True)
    num_models = len(queues)
    save_explr_update = True

    torch.set_num_threads(num_threads_real)
    killer = setup(rank,world_size,seed,args.use_gpu,skip_numa=use_async)

    test_traj = []

    # load variables
    if new_model_explr:
        model_dict,trainer_args = build_arg_dicts(args,replay_buffer)
        # create model and trainer
        if use_async:
            from vae import get_VAE
            VAE = get_VAE(args.learn_force)
            model = VAE(**trainer_args['model_dict']).to(device=args.device,dtype=args.dtype)
            model.device = args.device
            model.dtype = args.dtype
            optimizer = None
        else:
            from dist_modules.trainer_module import Trainer
            trainer = Trainer(trainer_args,0,killer)
            model = trainer.model
            optimizer = trainer.optimizer
        model.eval()
        model.build_chunk_decoder()
        weighted = True

        from dist_modules.sensor_main_module import SensorMain
        test = SensorMain(model,optimizer,replay_buffer,args,killer,explr_robot_lim_scale=1.25)
        test.xinit_orig = test.xinit.copy()
        test.use_vel = True
        test.extra_image_processing = lambda data: data[::args.extra_down_sample,::args.extra_down_sample,:]
        # test.num_traj_samples = test.num_traj_samples*2 # tighter exploration
        test.num_traj_samples = test.num_traj_samples//args.extra_down_sample # looser exploration
        test.robot.test(test.num_target_samples)
    else:
        from dist_modules.sensor_test_module import SensorTest,ExplrDist
        target_dist = ExplrDist(explr_idx=np.arange(len(explr_states)))
        target_dist.init = True
        target_dist.push(np.zeros(len(explr_states)),1.0)
        test = SensorTest(target_dist=target_dist,num_steps=args.num_steps,explr_states=explr_states,manual=False,init_vel=True)
        test.robot.test(test.num_target_samples)

    # give fingeprint test nodes time to intialize
    if not test.pybullet:
        rospy.sleep(10)
    test.rate = rospy.Rate(3.25) # slowed down to match fingerprint update rate
    # test.pause_pub.publish() # wait for fingerprint tests to be intialized before starting so buffers dont fill up

    # main loopnge
    fig1 = False
    done = False
    iter_step = 0
    explr_info = []
    ergodic_cost = []
    cprint(f'[TEST (RANK {rank})] starting main loop','cyan')
    while not rospy.is_shutdown() and not killer.kill_now and not done :
        while iter_step < args.num_steps+1:
            if test.got_state and test.got_img and (test.robot is not None) and not(test.pause):
                if not fig1 and test.use_vel and 'x' in test.states: # set up plotting
                    plot_zs = False
                    plot_queue.send(['explr',[test.plot_idx,test.xinit,test.dir_path,test.image_dim,plot_zs,test.robot_lim,test.tray_lim]])
                    fig1 = True
                success, out = test.step(iter_step)
                if success:
                    if new_model_explr:
                        out = test.data_buffer[-1] # need data without extra downsampling
                        if iter_step > test.frames_before_training:
                            if use_async:
                                test.load_model()
                                learning_ind = None
                            else:
                                # prep trainer
                                trainer.pre_train_mp(iter_step)
                                # do update
                                step_loss = trainer(weighted)
                                # update plots
                                trainer.post_train_mp(iter_step,step_loss)
                                learning_ind = trainer.learning_ind
                            # update plots
                            test.post_train_mp(iter_step,learning_ind)
                    else:
                        test.explr_update[0] = test.explr_update[0].T
                    fingerprint_buffer.push(*out,0,iter_step) # 0 is placholder for force
                    for idx,q in enumerate(queues): ## this line is slow
                        q.send(['update',iter_step])
                    if (tdist_queue is not None) and (iter_step > update_tdist_step-1) and (iter_step % update_rate == 0):
                        tdist = tdist_queue.recv()
                        explr_idx = 2
                        rospy.logwarn_once(f"tdist len: {len(tdist)}, exploring with tdist {explr_idx}")
                        test.robot.target_dist = tdist[explr_idx]
                    # update_figs
                    if fig1 and test.explr_update is not None:
                        plotter_buffer.push(*test.explr_update)
                        plot_queue.send(['explr_update',None]) # test.explr_update])
                        explr_info.append(test.explr_update)
                        ergodic_cost.append(get_cost(test.explr_update))                
                        test.explr_update = None
                    if fig1 and (iter_step % update_rate == 0 or video):
                        # save figs
                        for fpath in fpaths:
                            fname = fpath+f'_overview_step{iter_step:05d}.png' # .svg'
                            plot_queue.send(['save',[fname,iter_step]])
                    iter_step += 1
                if iter_step % update_rate == 0 :
                    cprint(f'[TEST (RANK {rank})] {iter_step}/{args.num_steps} | {test.duration_str} ','cyan')
                if iter_step > test.frames_before_training and iter_step % 50 == 0:
                    test.publish_distribution = True
            if not test.pybullet:
                test.rate.sleep()
        test.stop_pub.publish()

        if args.save_figs:
            # save traj
            test_traj = np.array(test_traj)
            for fpath in fpaths:
                fname = fpath+f'_overview_step{iter_step:05d}.png' # .svg'
                plot_queue.send(['save',[fname,iter_step]])
                pickle.dump( test_traj, open( fpath+"_explr_traj.pickle", "wb" ) )

        for q in queues:
            q.send(['done',None])
        cprint(f'[TEST (RANK {rank})]  done','cyan')
        done = True
    # close plotter
    with suppress(BrokenPipeError):
        plot_queue.send(['done',True])
    time.sleep(1)
    del plot_queue
    if tdist_queue is not None:
        del tdist_queue

    # save info
    if save_explr_update:
        pickle.dump( explr_info, open( test.dir_path+"explr_update_info.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump( ergodic_cost, open( test.dir_path+"ergodic_cost.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
    # cleanup process
    cprint('[TEST (RANK {rank})] shutdown','cyan')
