#!/usr/bin/env python

########## global imports ##########
from termcolor import cprint

########## local imports ##########
from dist_modules.utils import setup
import time
import pickle
import glob
from plotting.plotting_pyqtgraph import DebugPlotter
import torch 
import os
import numpy as np

throttle = False
resample = False

def debug_plotter(rank,world_size,seed,dir_path,states,image_dim,model,learn_force,shared_model=False,render=True,save_figs=True):

    killer = setup(rank,world_size,seed+5)

    # set up plotting
    from vae.vae_buffer import ReplayBufferTorch
    files = glob.glob(dir_path+'/*.pickle')
    if resample:
        vae_buffer = ReplayBufferTorch(capacity=len(files)*100,x_dim=len(states),y_dim=image_dim,device=model.device,dtype=model.dtype,learn_force=learn_force,world_size=1,batch_size=10)
    else: 
        test_x = []
        test_y = []
        if learn_force: 
            test_force = []
        else: 
            force = None
    for file in files: 
        if not('data_eval_dict' in file) and not('explr_update_info.pickle' in file): 
            with open(file,'rb') as f:
                test = pickle.load(f)
            try: 
                x = torch.as_tensor(test['state'],device=model.device,dtype=model.dtype)
                y = torch.as_tensor(test['data'],device=model.device,dtype=model.dtype)
                if learn_force: 
                    force = torch.as_tensor(test['force'],device=model.device,dtype=model.dtype)
                if resample:
                    vae_buffer.push_batch(x,y,force)
                else: 
                    random_indices = torch.randperm(x.shape[0])[:5]
                    test_x.append(x[random_indices])
                    test_y.append(y[random_indices])
                    if learn_force: 
                        test_force.append(force[random_indices])
            except: 
                cprint(f'[PLOTTER DEBUG] skipped {file}','green')                                

    if len(test_x) == 0: 
        cprint(f'[PLOTTER DEBUG {rank}] shutdown (no data)','green')
    else:

        if not render: 
            from pyvirtualdisplay import Display
            display = Display(visible=0, size=(1920,1080))
            display.start() 
            render=True

        if not resample:
            test_inputs = [torch.vstack(test_x),torch.vstack(test_y)]
            if learn_force: 
                test_inputs.append(torch.vstack(test_force))
            test_y = torch.vstack(test_y).permute(0,3,2,1).cpu().numpy()
        plot = DebugPlotter(render,dir_path+'/debug/',shared_model)

        cprint(f'[PLOTTER DEBUG {rank}] initialized','green')

        count = 0
        while not killer.kill_now:
            PATH=dir_path+'model_checkpoint_tmp.pth'
            if not(shared_model) and os.path.exists(PATH):
                try:
                    tmp = torch.load(PATH)
                    model.load_state_dict(tmp['model'],strict=False)
                except: 
                    print('failed to load model')
                    pass
            if resample and count % 10 == 0:
                test_inputs = vae_buffer.sample_batch()[:2]
                test_y = test_inputs[1].permute(0,3,2,1).cpu().numpy()
            with torch.no_grad():
                img_pred = model(*test_inputs)[0].permute(0,3,2,1).cpu().numpy()
                if shared_model:
                    if model.init: 
                        seed = model.seed_y.clone().permute(0,3,2,1)
                        img_pred_seeded = model.decode_samples_only(test_inputs[0],get_pred=True)[0].permute(0,3,2,1).cpu().numpy()
                    else: 
                        seed = np.zeros_like(img_pred[[0]])
                        img_pred_seeded = img_pred
                else: 
                    img_pred_seeded = None
            plot.update(test_y,img_pred,seed,img_pred_seeded) 
            if save_figs and (count > 10) and (count % 100 == 0): 
                try:
                    plot.save(f'debug_{count}')
                except: 
                    print('saving debug plot failed')
            count += 1
            time.sleep(3)

        cprint('[PLOTTER DEBUG] shutdown','green')
