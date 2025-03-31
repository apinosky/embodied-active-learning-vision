#!/usr/bin/env python

########## global imports ##########
from termcolor import cprint

########## local imports ##########
from dist_modules.utils import setup
import time
import numpy as np

throttle = False

def plotter(rank,world_size,seed,plot_queue,states,plotter_buffer=None,render=True,save_figs=True):
    killer = setup(rank,world_size,seed)
    fig1 = None
    done = False
    display = None

    if not render: 
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1920,1080))
        display.start() 
        render=True
        throttle=True

    last_recv_time = time.time()
    train = False
    timeout = 0.1
    count = 0
    save_iters = []
    while not killer.kill_now and not done:
        try:
            if plot_queue.poll(timeout=timeout):
                # print(plot_queue.qsize())
                out = plot_queue.recv()
                if not isinstance(out[0],list): # if type is not "many"
                    out = [out]
                for plt_type,data in out:
                    last_recv_time = time.time()
                    if not fig1: # set up plotting
                        # print(xinit)
                        if plt_type == 'explr':
                            plot_idx,xinit,dir_path,image_dim,plot_zs,robot_lim,tray_lim = data
                            from plotting.plotting_pyqtgraph import Plotter
                            # from plotting.plotting_matplotlib import Plotter
                            # from plotting.plotting_matplotlib import Plotter3D as Plotter
                            plot = Plotter(plot_idx,xinit,path=dir_path,plot_zs=plot_zs,states=states,render=render,robot_lim=robot_lim, tray_lim=tray_lim)
                            timeout = 0.01
                        elif plt_type == 'train' :
                            dir_path,image_dim = data
                            # from plotting.plotting_matplotlib import TrainingPlotter
                            from plotting.plotting_pyqtgraph import TrainingPlotter
                            plot = TrainingPlotter(path=dir_path,render=render)
                            train = True
                        fig1 = True
                        cprint('[PLOTTER {} {}] initialized'.format('(train)' if train else '(explr)',rank),'green')
                    else:
                        if plt_type=='explr_update':
                            if plotter_buffer is not None:
                                ready = len(plotter_buffer) > 0
                                while not ready and not done:
                                    time.sleep(0.05)
                                    ready = len(plotter_buffer) > 0
                                data = plotter_buffer.get_next()
                            data[0] = data[0].cpu().numpy()
                            data[1] = data[1].cpu().numpy()
                            data[2] = data[2].cpu().numpy()
                            if plot_zs:
                                for idx in [4,5,6]:
                                    data[idx] = data[idx].cpu().squeeze().numpy()
                            if len(save_iters) > 0: 
                                iter_step = data[-1][0]-1
                                if save_iters[0] == iter_step: 
                                    plot.save(f'explr_step{iter_step:05d}')
                                    save_iters.pop(0)
                                elif iter_step > save_iters[0]:
                                    plot.save(f'explr_step{iter_step:05d}')
                                    save_iters.pop(0)
                            iter_step = data[-1][0]
                            if (display is None): 
                                throttle = len(out) > 10
                            plot.update(data,throttle=(throttle and (iter_step % 10 != 0))) #,draw=(count % 10 == 0))
                            # count += 1
                            if len(save_iters) > 0: 
                                if save_iters[0] == iter_step: 
                                    plot.save(f'explr_step{iter_step:05d}')
                                    save_iters.pop(0)
                        elif plt_type=='training_update':
                            for idx in [0,1,2]:
                                data[idx] = data[idx].cpu().numpy()
                            plot.training_update(data)
                        elif plt_type=='checkpoint_update':
                            for idx in [0,1,2]:
                                data[idx] = data[idx].cpu().numpy()
                            plot.checkpoint_update(data)
                        elif plt_type=='save' and save_figs:
                            fname,iter_step = data
                            if plotter_buffer is not None:
                                save_iters.append(iter_step)
                            else:
                                if (not train) and (fname is not None):
                                    plot.save(fname,full_path=True)
                                else:
                                    plot.save(f'explr_step{iter_step:05d}')
                        elif plt_type=='save_fig3_only' and save_figs:
                            plot.save_fig3_only(data)
                        elif plt_type=='done':
                            done = True
                    # del plot_data
            elif (time.time() - last_recv_time) > 60*60:
                done = True
                cprint('[PLOTTER] timed-out','green')
        except (BrokenPipeError,EOFError):
            break
    if fig1:
        plot.save('{}final'.format('' if train else 'explr_'))

    del plot_queue
    del plot
    cprint('[PLOTTER {}] shutdown'.format('(train)' if train else '(explr)'),'green')
