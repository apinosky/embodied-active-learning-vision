#!/usr/bin/env python

########## global imports ##########
import torch
import numpy as np
import rospy

from termcolor import cprint
import datetime
import time
from contextlib import suppress
import pickle
import signal

########## local imports ##########
from .utils import setup, cleanup, get_num_cores
from .sensor_main_module import SensorMain
from .sensor_utils import get_cost, numba_handler

video=False

def main(rank,world_size,args,trainer_args,plot_queue,num_threads_real,plotter_buffer=None,clustering_queue=None):
    torch.set_num_threads(num_threads_real) # 1)
    killer = setup(rank,world_size,args.seed,trainer_args['use_gpu'],skip_numa=False)
    # args.device='cpu'
    # build model
    from vae import get_VAE
    VAE = get_VAE(args.learn_force)
    model = VAE(**trainer_args['model_dict']).to(device=args.device,dtype=args.dtype)
    model.device = args.device
    model.dtype = args.dtype
    model.eval()
    num_threads = get_num_cores()
    possible_chunks=np.arange(1,num_threads+1)
    chunks=int(possible_chunks[(args.num_target_samples % possible_chunks) == 0][-1] )
    model.build_chunk_decoder(chunks)
    # model.build_z_buffer()
    replay_buffer = trainer_args['replay_buffer']
    cprint(f'[MASTER {rank}] initialized','cyan')
    if 'shared_model' in trainer_args.keys():
        shared_model = True     
    else:
        shared_model = False
        trainer_args['shared_model'] = None
    test = SensorMain(model,None,replay_buffer,args,killer,shared_model=trainer_args['shared_model'])
    # check numba config
    signal.signal(signal.SIGALRM, numba_handler)
    try:
        signal.alarm(2)
        test.robot.test(test.num_target_samples)
    except Exception:
        raise TimeoutError('numba got stuck, try disabling parallel in control/klerg_utils.py or setting use_numba=False in control/kerg.py')
    signal.alarm(0)

    # main loop
    fig1 = False
    done = False
    iter_step = 0
    explr_info = []
    ergodic_cost = []
    ### set up safe shutdown
    while not rospy.is_shutdown() and not done and not killer.kill_now:
        while iter_step < test.num_steps*test.data_to_ctrl_rate and not killer.kill_now:
            if test.got_state and test.got_img and (test.robot is not None) and not(test.pause):
                if (not fig1) and (plot_queue is not None): # set up plotting
                    plot_zs = True
                    robot_lim = test.robot_lim
                    tray_lim = test.tray_lim
                    if test.use_magnitude: 
                        robot_lim[test.robot.vel_locs,0] = 0.
                        tray_lim[test.robot.vel_locs,0] = 0.                        
                    plot_queue.send(['explr',[test.plot_idx,test.xinit,test.dir_path,test.image_dim,plot_zs,robot_lim,tray_lim]])
                    fig1 = True
                start = time.time()
                ratio = test.learning_ind/max(1,iter_step-test.frames_before_training)
                move_only = False # (ratio < test.target_learning_rate ) and (iter_step > test.frames_before_training)
                success, out = test.step(iter_step,move_only=move_only)
                test.load_model(shared_model=shared_model)
                if success:
                    # check for VAE update (multiprocessing)
                    if iter_step > test.frames_before_training:
                        test.post_train_mp(iter_step)
                        if args.print_debug: cprint(f'[MASTER] finished update | {time.time()-start}','cyan')
                    if iter_step > 0 and iter_step % 1000 == 0:
                        test.write_to_log(f'saving intermediate model @ {iter_step} steps')
                        test.save(mod=f'_{iter_step}steps') #,save_model=(iter_step%1000==0))
                    # update_figs
                    if test.explr_update is not None:
                        if plot_queue is not None:
                            if plotter_buffer is None:
                                plot_queue.send(['explr_update',test.explr_update])
                            else:
                                plotter_buffer.push(*test.explr_update)
                                plot_queue.send(['explr_update',None])
                        explr_info.append(test.explr_update)
                        ergodic_cost.append(get_cost(test.explr_update))
                        test.explr_update = None
                    if iter_step > test.frames_before_training and iter_step % 50 == 0:
                        if clustering_queue is not None:
                            test.save_clustering_model(shared_model)
                            clustering_queue.send([iter_step,False, (iter_step % 500 == 0)])
                        else: 
                            test.publish_distribution = True # for clustering
                    # intermediate save
                    if ((iter_step % 10 == 0) or video) and (plot_queue is not None):
                        plot_queue.send(['save',[None,iter_step]])
                    iter_step += 1
            # pre = time.time()
            if not test.pybullet:
                test.rate.sleep()
            # print('sleep',time.time()-pre)

        test.save(post_explr=False)
        done = True

    done = False
    while not rospy.is_shutdown() and not done and not killer.kill_now:
        while test.learning_ind < test.num_steps*test.target_learning_rate and not killer.kill_now:
            if test.got_state and test.got_img and (test.robot is not None) and not(test.pause):
                start = time.time()
                success, out = test.step(iter_step)
                if success:
                    # check for VAE update (multiprocessing)
                    if iter_step > test.frames_before_training:
                        test.load_model(shared_model=shared_model)
                        test.post_train_mp(iter_step)
                        if args.print_debug: cprint(f'[MASTER] finished update | {time.time()-start}','cyan')
                    if iter_step > 0 and iter_step % 1000 == 0:
                        test.write_to_log(f'saving intermediate model @ {iter_step} steps')
                        test.save(mod=f'_{iter_step}steps') #,save_model=(iter_step%1000==0))
                    # update_figs
                    if test.explr_update is not None:
                        if plot_queue is not None:
                            if plotter_buffer is None:
                                plot_queue.send(['explr_update',test.explr_update])
                            else:
                                plotter_buffer.push(*test.explr_update)
                                plot_queue.send(['explr_update',None])
                        explr_info.append(test.explr_update)
                        ergodic_cost.append(get_cost(test.explr_update))
                        test.explr_update = None
                    if iter_step > test.frames_before_training and iter_step % 250 == 0:
                        if clustering_queue is not None:
                            test.save_clustering_model(shared_model)
                            clustering_queue.send([iter_step,False, (iter_step % 500 == 0) or video])
                        else: 
                            test.publish_distribution = True # for clustering
                    # intermediate save
                    if ((iter_step % 10 == 0) or video) and (plot_queue is not None):
                        plot_queue.send(['save',[None,iter_step]])
                    iter_step += 1
            # pre = time.time()
            if not test.pybullet:
                test.rate.sleep()
            # print('sleep',time.time()-pre)

        test.save(post_explr=True)
        done = True

    test.stop_pub.publish()
    # close plotter
    if (plot_queue is not None):
        with suppress(BrokenPipeError):
            plot_queue.send(['done',True])
    # close clustering
    if clustering_queue is not None:
        with suppress(BrokenPipeError):
            clustering_queue.send([iter_step,True,True])
    time.sleep(1)
    del plot_queue
    del clustering_queue
    # save info
    pickle.dump( explr_info, open( test.dir_path+"explr_update_info.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump( ergodic_cost, open( test.dir_path+"ergodic_cost.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
    # cleanup process
    cprint('[MASTER] shutdown','cyan')
