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
import matplotlib.pyplot as plt

########## local imports ##########
import sys, os
from .utils import setup, cleanup, get_num_cores
from .sensor_main_module import SensorMain
from .sensor_utils import get_cost, numba_handler


def main(rank,world_size,args,trainer_args,explr_plot_queue,train_plot_queues,data_queues,plotter_buffer=None,clustering_queue=None):
    if isinstance(train_plot_queues,list):
        train_plot_queue = train_plot_queues[0]

    if args.ddp_trainer:
        import torch.distributed as dist
        from .trainer_ddp import setup_trainer
        replay_buffer = trainer_args['replay_buffer']
    else:
        new_path = os.path.dirname(os.path.abspath(__file__))+"/../."
        sys.path.append(new_path)
        from vae.vae_buffer import ReplayBufferTorch
        from .trainer_module import Trainer

        replay_buffer = ReplayBufferTorch(capacity=args.buffer_capacity,x_dim=args.s_dim,y_dim=args.image_dim,device=args.device,dtype=args.dtype,learn_force=args.learn_force,world_size=1,batch_size=args.batch_size)
        trainer_args['replay_buffer'] = replay_buffer

    killer = setup(rank,world_size,args.seed,trainer_args['use_gpu'])
    if args.ddp_trainer: 
        trainer = setup_trainer(rank,world_size,trainer_args,killer)
    else:
        trainer = Trainer(trainer_args,rank,killer)
    cprint(f'[MASTER {rank}] DDP initialized','cyan')
    if 'shared_model' in trainer_args.keys():
        shared_model = True     
    else:
        shared_model = False
        trainer_args['shared_model'] = None

    test = SensorMain(trainer.model,trainer.optimizer,replay_buffer,args,killer,shared_model=trainer_args['shared_model'])
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
    weighted = True

    ### set up loss figs
    from .trainer_module import update_loss_plots
    loss_fig = None
    plot_losses = False
    plot_combo = True

    ### set up safe shutdown
    while not rospy.is_shutdown() and not done and not killer.kill_now:
        while iter_step < test.num_steps*test.data_to_ctrl_rate and not killer.kill_now: 
            if test.got_state and test.got_img and (test.robot is not None) and not(test.pause):
                if (not fig1): # set up plotting
                    if (explr_plot_queue is not None):
                        plot_zs = True
                        robot_lim = test.robot_lim
                        tray_lim = test.tray_lim
                        if test.use_magnitude: 
                            robot_lim[test.robot.vel_locs,0] = 0.
                            tray_lim[test.robot.vel_locs,0] = 0.                        
                        explr_plot_queue.send(['explr',[test.plot_idx,test.xinit,test.dir_path,test.image_dim,plot_zs,robot_lim,tray_lim]])
                    if (train_plot_queue is not None):
                        [p.send(['train',[test.dir_path,test.image_dim]]) for p in train_plot_queues]
                    fig1 = True
                start = time.time()
                success, out = test.step(iter_step)
                if success:
                    # do VAE update (multiprocessing)
                    if iter_step > test.frames_before_training:
                        # prep trainer
                        trainer.pre_train_mp(iter_step)
                        try:
                            for dq in data_queues:
                                dq.send([weighted,False])
                            if args.print_debug: cprint('[MASTER] data sent to all trainers','cyan')
                            step_loss = trainer(weighted)
                            if args.ddp_trainer:
                                get_loss = dist.reduce(step_loss,dst=0,op=dist.ReduceOp.SUM, async_op=True)
                                get_loss.wait()
                                step_loss /= world_size                                
                        except (BrokenPipeError) as c:
                            # cprint(f'[MASTER {rank}] ERROR {c}','cyan')
                            break
                        except (RuntimeError) as c:
                            cprint(f'[MASTER {rank}] ERROR {c}','cyan')
                            break                        # gather loses
                        if args.print_debug: cprint(f'[MASTER] {step_loss}','cyan')
                        test.post_train_mp(iter_step,trainer.learning_ind)
                        # update plots
                        trainer.post_train_mp(iter_step,step_loss,plot=(train_plot_queue is not None ))
                        loss_fig = update_loss_plots(plot_losses,plot_combo,trainer,loss_fig)
                        if args.print_debug: cprint(f'[MASTER] finished update | {time.time()-start}','cyan')
                    iter_step += 1
                    if iter_step > 0 and iter_step % 1000 == 0:
                        test.write_to_log(f'saving intermediate model @ {iter_step} steps')
                        test.save(mod=f'_{iter_step}steps',losses=trainer.losses.copy())
                    # update_figs
                    if test.explr_update is not None:
                        if explr_plot_queue is not None:
                            if plotter_buffer is None:
                                explr_plot_queue.send(['explr_update',test.explr_update])
                            else:
                                plotter_buffer.push(*test.explr_update)
                                explr_plot_queue.send(['explr_update',None])
                        explr_info.append(test.explr_update)
                        ergodic_cost.append(get_cost(test.explr_update))
                        test.explr_update = None
                    if train_plot_queue is not None:
                        if trainer.training_update is not None:
                            train_plot_queue.send(['training_update',trainer.training_update])
                            trainer.training_update = None
                        if trainer.checkpoint_update is not None:
                            train_plot_queue.send(['checkpoint_update',trainer.checkpoint_update])
                            trainer.checkpoint_update = None
                    if iter_step > test.frames_before_training and iter_step % 50 == 0:
                        if clustering_queue is not None:
                            test.save_clustering_model(shared_model)
                            clustering_queue.send([iter_step,False,iter_step % 500==0])
                        else: 
                            test.publish_distribution = True # for clustering
                    # intermediate save
                    if args.save_figs and (explr_plot_queue is not None):
                        if iter_step % 10 == 0 :
                            explr_plot_queue.send(['save',[f'step{iter_step-1:05d}',iter_step-1]])
                            train_plot_queue.send(['save',[f'step{iter_step-1:05d}',iter_step-1]])
                        else:
                            explr_plot_queue.send(['save',[None,iter_step-1]])
            if not test.pybullet:
                test.rate.sleep()

        test.save(post_explr=False,losses=trainer.losses.copy())

        test.write_to_log('done')
        done = True
    if plot_losses:
        if len(loss_fig[0]) > 0:
            loss_fig[0][0].savefig(trainer.dir_path+f'/trainer_trends_step{trainer.learning_ind:05d}.svg')

    test.stop_pub.publish()
    # close loss figs
    if plot_losses:
        [plt.close(lf[0]) for lf in loss_fig if len(lf) > 0]

    # close trainers
    for dq in data_queues:
        with suppress(BrokenPipeError):
            dq.send([True,True])
    # close plotter
    if explr_plot_queue is not None:
        with suppress(BrokenPipeError):
            explr_plot_queue.send(['done',True])
            for p in train_plot_queues: 
                p.send(['done',True])
    # close clustering
    if clustering_queue is not None:
        with suppress(BrokenPipeError):
            clustering_queue.send([iter_step,True,True])
    time.sleep(1)
    del explr_plot_queue
    del clustering_queue
    # save info
    pickle.dump( explr_info, open( test.dir_path+"explr_update_info.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump( ergodic_cost, open( test.dir_path+"ergodic_cost.pickle", "wb" ),protocol=pickle.HIGHEST_PROTOCOL)
    # cleanup process
    cleanup()
    cprint(f'[Trainer {rank}] shutdown','yellow')
    cprint('[MASTER] shutdown','cyan')
