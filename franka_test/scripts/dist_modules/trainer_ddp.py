#!/usr/bin/env python

########## global imports ##########
import torch
import torch.distributed as dist

from termcolor import cprint
import datetime
import time
from contextlib import suppress
import matplotlib.pyplot as plt
import numpy as np

########## local imports ##########
from .utils import setup, cleanup, set_seeds, get_num_cores
from .trainer_module import Trainer
from plotting.plotting_matplotlib import set_mpl_format

set_mpl_format()
video=False 

def setup_trainer(rank,world_size,trainer_args,killer):
    use_gpu = trainer_args['args'].use_gpu
    if not use_gpu: 
        try:
            dist.init_process_group(
                    backend="ccl", rank=rank, world_size=world_size,init_method="tcp://localhost:29500",
                    timeout=datetime.timedelta(seconds=900)
                )
        except ValueError:
            if dist.is_gloo_available():
                # if torch.cuda.is_available() and use_gpu and dist.is_gloo_available():
                dist.init_process_group(
                        backend="gloo", rank=rank, world_size=world_size,init_method="tcp://localhost:29500",
                        timeout=datetime.timedelta(seconds=900)
                    )
                print("ccl unavailable, using gloo backend")
            else:
                raise ValueError("cpu requested and ccl and gloo backends unavailable")
    else: 
        if torch.cuda.is_available() and use_gpu and dist.is_nccl_available():
            dist.init_process_group(
                    backend="nccl", rank=rank, world_size=world_size,init_method="tcp://localhost:29500",
                    timeout=datetime.timedelta(seconds=900)
                )
        elif dist.is_gloo_available() :
            # if torch.cuda.is_available() and use_gpu and dist.is_gloo_available():
            dist.init_process_group(
                    backend="gloo", rank=rank, world_size=world_size,init_method="tcp://localhost:29500",
                    timeout=datetime.timedelta(seconds=900)
                )
            print("nccl unavailable, using gloo backend")
        else:
            raise ValueError("cuda requested and nccl and gloo backends unavailable")
    cprint(f'[TRAINER {rank}] DDP initialized','yellow')
    return Trainer(trainer_args,rank,killer)

def train(rank,world_size,args,seed,data_queue):
    killer = setup(rank,world_size,seed,args['use_gpu'])
    if args['use_gpu']: 
        args['args'].device = f'cuda:{rank}'
    trainer = setup_trainer(rank,world_size,args,killer)
    done = False
    while not killer.kill_now and not done:
        try:
            if data_queue.poll(timeout=1):
                weighted,done = data_queue.recv()
                if done:
                    cprint(f'[TRAINER {rank}] got done','yellow')
                    break
                if trainer.print_debug: cprint(f'[TRAINER {rank}] got batch data','yellow')
                step_loss = trainer(weighted)
                dist.reduce(step_loss,dst=0,op=dist.ReduceOp.SUM)
                # dist.reduce(step_loss,dst=0,op=dist.ReduceOp.SUM,async_op=True)
                if trainer.print_debug: cprint(f'[TRAINER {rank}] sent loss data','yellow')
        except (BrokenPipeError,RuntimeError) as c:
            # cprint(f'[TRAINER {rank}] ERROR {c}','yellow')
            break
    cleanup()
    cprint(f'[TRAINER {rank}] shutdown','yellow')

def train_async(rank,world_size,args,seed,plot_queues,data_queues,ddp_trainer=True,plot_losses=False,plot_combo=True):
    if isinstance(plot_queues,list):
        plot_queue = plot_queues[0]
    else: 
        plot_queue = plot_queues
    killer = setup(rank,world_size,seed,args['use_gpu'])
    if args['use_gpu']: 
        args['args'].device = f'cuda:{rank}'
    if ddp_trainer:
        trainer = setup_trainer(rank,world_size,args,killer)
    else:
        trainer = Trainer(args,rank,killer)
    num_threads = get_num_cores()
    possible_chunks=np.arange(1,num_threads+1)
    chunks=int(possible_chunks[(trainer.num_target_samples % possible_chunks) == 0][-1] )
    trainer.build_chunk_decoder(chunks)
    # main loop
    done = False
    fig1 = False
    save_checkpoints = False
    last_save = 0
    save_freq = 500
    save_step = save_freq
    if trainer.print_debug: cprint(f'check if initialized {dist.is_initialized()}','red')
    if trainer.print_debug: cprint(f'check if backend {dist.get_backend()}','red')
    if trainer.print_debug: cprint(f'check group {dist.group.WORLD}','red')
    # print('creating trainer node')
    cprint(f'[Trainer {rank}] main loop','green')

    ### set up loss figs
    from .trainer_module import update_loss_plots
    loss_fig = None
    unweighted_count = 0.

    ### set up safe shutdown
    while trainer.replay_buffer.explr_step < trainer.num_steps and not killer.kill_now:
        # do VAE update (multiprocessing)
        explr_step = trainer.replay_buffer.explr_step
        ratio = trainer.get_learning_ratio(trainer.learning_ind-unweighted_count,explr_step)

        ### uncomment below if you want trainer to wait for exploration to collect enough samples to maintain desired rate (always do weighted sampling)
        weighted = False 
        ok_ratio = ratio < trainer.target_learning_rate 

        ### uncomment below if you want trainer to do uniform sampling when trainer is faster than exploration (and keep track of unweighted learning steps)            
        # weighted = ratio < trainer.target_learning_rate 
        # ratio2 = trainer.get_learning_ratio(trainer.learning_ind,explr_step)
        # ok_ratio = ratio2 < trainer.target_learning_rate*2

        if (not trainer.replay_buffer.paused) and ok_ratio and explr_step >= trainer.frames_before_training:
            # print(iter_step+trainer.frames_before_training,explr_step)
            if (plot_queue is not None) and (not fig1): # set up plotting
                [p.send(['train',[trainer.dir_path,trainer.image_dim]]) for p in plot_queues]
                fig1 = True
            # prep trainer
            trainer.pre_train_mp(explr_step)
            # do multiprocessing update
            if ddp_trainer:
                try:
                    for dq in data_queues:
                        dq.send([weighted,False])
                    if trainer.print_debug: cprint(f'[TRAINER {rank}] data sent to all trainers','cyan')
                    start = time.time()
                    step_loss = trainer(weighted)
                    # single_time = time.time()-start
                    # gather loses
                    get_loss = dist.reduce(step_loss,dst=0,op=dist.ReduceOp.SUM, async_op=True)
                    get_loss.wait()
                    step_loss /= world_size
                except (BrokenPipeError) as c:
                    # cprint(f'[TRAINER {rank}] ERROR {c}','cyan')
                    break
                except (RuntimeError) as c:
                    if not killer.kill_now:
                        cprint(f'[TRAINER {rank}] ERROR {c}','cyan')
                    break
                if trainer.print_debug: cprint(f'[TRAINER {rank}] {step_loss}','cyan')
            else: 
                step_loss = trainer(weighted)
            # update plots
            # if not weighted: 
            #     unweighted_count += len(step_loss)
            trainer.post_train_mp(explr_step,step_loss,plot=(plot_queue is not None )) #,unweighted_count=unweighted_count)
            loss_fig = update_loss_plots(plot_losses,plot_combo,trainer,loss_fig)

            if trainer.print_debug: cprint(f'[TRAINER {rank}] finished update | {time.time()-start}','cyan')
            trainer.save_checkpoint()
            # print('saved model')

            explr_step = trainer.replay_buffer.explr_step
            if (explr_step > last_save) and (save_step <= explr_step):
                if save_checkpoints:
                    trainer.write_to_log(f'saving intermediate model @ {save_step} steps')
                    trainer.save(mod=f'_trainer_{save_step}steps')
                    last_save = explr_step
                    save_step += save_freq

            # update_figs
            if plot_queue is not None: 
                if trainer.training_update is not None:
                    plot_queue.send(['training_update',trainer.training_update])
                    trainer.training_update = None
                    if video:
                        plot_queue.send(['save',[f'step{trainer.learning_ind:05d}',trainer.learning_ind]])
                if trainer.checkpoint_update is not None:
                    plot_queue.send(['checkpoint_update',trainer.checkpoint_update])
                    trainer.checkpoint_update = None
                    plot_queue.send(['save',[f'step{trainer.learning_ind:05d}',trainer.learning_ind]])
            # print('TRAINER',single_time)
        else:
            time.sleep(0.1)
    trainer.save(post_explr=False)
    trainer.write_to_log('done w/ exploration training')
    if loss_fig is not None:
        if len(loss_fig[0]) > 0:
            loss_fig[0][0].savefig(trainer.dir_path+f'/trainer_trends_step{trainer.learning_ind:05d}.svg')
        # [plt.close(lf[0]) for lf in loss_fig if len(lf) > 0]

    # post explr training
    # weighted = False
    while trainer.learning_ind <= trainer.num_steps*trainer.target_learning_rate and not killer.kill_now:
        explr_step = trainer.replay_buffer.explr_step
        # prep trainer
        trainer.pre_train_mp(explr_step) #,last=False)
        # do multiprocessing update
        if ddp_trainer:
            try:
                for dq in data_queues:
                    dq.send([weighted,False])
                if trainer.print_debug: cprint(f'[TRAINER {rank}] data sent to all trainers','cyan')
                start = time.time()
                step_loss = trainer(weighted)
                # single_time = time.time()-start
                get_loss = dist.reduce(step_loss,dst=0,op=dist.ReduceOp.SUM, async_op=True)
                get_loss.wait()
                step_loss /= world_size
            except (BrokenPipeError) as c:
                # cprint(f'[TRAINER {rank}] ERROR {c}','cyan')
                break
            except (RuntimeError) as c:
                cprint(f'[TRAINER {rank}] ERROR {c}','cyan')
                break
            # gather loses
            if trainer.print_debug: cprint(f'[TRAINER {rank}] {step_loss}','cyan')
        else:
            step_loss = trainer(weighted)
        # update plots
        trainer.post_train_mp(explr_step,step_loss,plot=(plot_queue is not None )) #,last=False)
        loss_fig = update_loss_plots(plot_losses,plot_combo,trainer,loss_fig)

        if trainer.print_debug: cprint(f'[TRAINER {rank}] finished update | {time.time()-start}','cyan')
        trainer.save_checkpoint()
        # print('saved model')

        # update_figs
        if plot_queue is not None: 
            if trainer.training_update is not None:
                plot_queue.send(['training_update',trainer.training_update])
                trainer.training_update = None
            if trainer.checkpoint_update is not None:
                plot_queue.send(['checkpoint_update',trainer.checkpoint_update])
                trainer.checkpoint_update = None
                plot_queue.send(['save',[f'step{trainer.learning_ind:05d}',trainer.learning_ind]])

    if loss_fig is not None:
        if len(loss_fig[0]) > 0:
            loss_fig[0][0].savefig(trainer.dir_path+f'/trainer_trends_step{trainer.learning_ind:05d}.svg')
    trainer.save(post_explr=True)
    trainer.write_to_log('done w/ post-exploration training')

    # close figures 
    if loss_fig is not None:
        [plt.close(lf[0]) for lf in loss_fig if len(lf) > 0]
    
    # close trainers
    for dq in data_queues:
        with suppress(BrokenPipeError):
            dq.send([True,True])
    # close plotter
    if plot_queue is not None:
        for p in plot_queues:
            with suppress(BrokenPipeError):
                p.send(['done',True])
    # cleanup process
    cleanup()
    del plot_queue
    cprint(f'[Trainer {rank}] shutdown','yellow')
