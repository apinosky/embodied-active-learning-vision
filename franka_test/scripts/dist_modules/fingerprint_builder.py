#!/usr/bin/env python

########## global imports ##########
import numpy as np
import matplotlib.pyplot as plt
import pickle
import yaml

from termcolor import cprint

import torch

########## local imports ##########
from franka.franka_utils import ws_conversion
from control.klerg_utils import *
from .utils import set_seeds, get_num_cores, add_yaml_representers
from .clustering import find_clusters,relabel, plot_gmm_results

add_yaml_representers()

class FingerprintBuilder(object):
    def __init__(self,dir_path='.',model_path='model_final.pth',buffer_name='explr_update_info.pickle',buffer=None,model=None):

        # load variables
        with open(dir_path + "/config.yaml","r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.map_dict(params)
        self.num_pts = 1000 # was 5000
        self.dir_path = dir_path
        # self.render_figs=True # manual override for debugging

        # seeds
        set_seeds(self.seed)

        if model is None: 
            ## Initialize VAE
            loaded_info = torch.load(self.dir_path + model_path)
            if not isinstance(loaded_info,dict):
                ## raw model load
                self.model = loaded_info.to('cpu')
            else:
                ## load state_dict only
                from dist_modules.utils import build_arg_dicts
                from vae import get_VAE
                VAE = get_VAE(self.learn_force)
                model_dict,_ = build_arg_dicts(self,None)
                self.model = VAE(**model_dict).to('cpu')
                self.model.load_state_dict(loaded_info,strict=False)
            # print(self.model)
            self.model.eval()

            # torch.set_num_threads(1)
            num_threads = get_num_cores()
            possible_chunks=np.arange(1,num_threads+1)
            chunks=int(possible_chunks[(self.num_pts % possible_chunks) == 0][-1] )
            chunks = min(10,chunks)
            self.model.build_chunk_decoder(chunks)
        else: 
            self.model = model

        self.reshape = lambda y: np.clip(y,0,1)

        # initalize buffers
        self.tray_lim = np.array(self.tray_lim)
        self.robot_lim = np.array(self.robot_lim)
        self.tray_ctrl_lim = np.array(self.tray_ctrl_lim)
        self.robot_ctrl_lim = np.array(self.robot_ctrl_lim)
        self.dtype = eval(self.dtype)

        self.batch_size = 10

        # load replay buffer
        if buffer is None:
            with open(self.dir_path + "/" + buffer_name ,"rb") as f:
                data_dict = pickle.load(f)
            from vae.vae_buffer import ReplayBufferTorch
            self.vae_buffer = ReplayBufferTorch(capacity=len(data_dict),x_dim=self.s_dim,y_dim=self.image_dim,device=self.device,dtype=self.dtype,learn_force=self.learn_force,world_size=1,batch_size=self.batch_size)
            batch = [[d[1],d[0],d[2]] for d in data_dict]
            x, y, force  = map(torch.stack, zip(*batch))
            self.vae_buffer.push_batch(x,y.permute(0,3,2,1),force=force)
        else:
            self.vae_buffer = buffer

    @torch.no_grad()
    def init_model(self,seed_x,seed_y,seed_force=None):
        seed_x = torch.FloatTensor(seed_x).unsqueeze(axis=0)
        seed_y = torch.FloatTensor(seed_y).unsqueeze(axis=0).permute(0,3,2,1)
        if self.model.learn_force:
            seed_force = torch.FloatTensor(seed_force).unsqueeze(axis=0)
            out = self.model.update_dist(seed_x, seed_y, seed_force)
        else:
            out = self.model.update_dist(seed_x, seed_y)
        seed_img_pred, z_mu, z_logvar = out[0],out[2],out[3]
        return seed_img_pred.detach().permute(0,3,2,1).numpy().squeeze(), [z_mu.detach().numpy().squeeze(), z_logvar.exp().detach().numpy().squeeze()]

    def get_prediction(self,x):
        img_pred,img_logvar = self.model.decode_samples_only(x,get_pred = True)
        return img_pred

    def map_dict(self, user_info):
        for k, v in user_info.items():
            setattr(self, k, v)

    def find_clusters(self,num_fingerprints,save_name=None,
        visualize_test_batches=False,get_blank=True,cluster_by_plot_idx=False,
        sample_method='reweight',cluster_method='mean_shift',optimize_samples=True):
        
        figs = []
        if self.batch_size > 0: 
            if visualize_test_batches and self.render_figs:
                figs_ok = 'N'
                while 'n' in figs_ok.lower():
                    # pull images to test
                    out = self.vae_buffer.sample(self.batch_size)
                    if self.model.learn_force:
                        out = out[:3]
                        test_xs,test_ys,test_forces = out
                    else: 
                        out = out[:2]
                        test_xs,test_ys = out[:2]
                    batch_fig,axs = plt.subplots(1,self.batch_size,figsize=(self.batch_size*3,1*3))
                    for y,ax in zip(test_ys,axs.flatten() ):
                        ax.imshow(y.cpu().numpy().T)
                    plt.show(block=False)
                    figs_ok = input('Do the figures look ok? Y/N (N will select new figures) ')
                    plt.close(batch_fig)
            else:
                # pull images to test
                out = self.vae_buffer.sample(self.batch_size)
                if self.model.learn_force:
                    out = out[:3]
                    test_xs,test_ys,test_forces = out
                else:
                    out = out[:2]
                    test_xs,test_ys = out
                batch_fig,axs = plt.subplots(1,self.batch_size,figsize=(self.batch_size*1.5,1.5))
                for y,ax in zip(test_ys,axs.flatten() ):
                    ax.imshow(y.cpu().numpy().T)
                    ax.xaxis.set_ticklabels([])
                    ax.xaxis.set_ticks([])
                    ax.yaxis.set_ticklabels([])
                    ax.yaxis.set_ticks([])
                batch_fig.tight_layout()
                if self.render_figs:
                    plt.show(block=False)
                figs.append(batch_fig)
        else: 
            out = None

        ## find clusters ##
        plot_idx = [self.states.rfind(s) for s in self.plot_states]

        clustering_config = {
            'model': self.model,
            'states': self.states,
            'plot_idx': plot_idx,
            'tray_lim': self.tray_lim,
            'robot_lim': self.robot_lim,
            'robot_ctrl_lim': self.robot_ctrl_lim,
            'num_pts': self.num_pts,
            'scale': 1.3, # 1.15,
            'uniform': True,
            'print_debug': True,
            'num_fingerprints':num_fingerprints,           # not used if cluster_method = mean_shift
            'get_blank': get_blank,
            'cluster_by_plot_idx': False,    # not used if optimize_samples = True
            'cluster_weight': False,
            'batch': out,
            'sample_method': sample_method,
            'cluster_method': cluster_method,
            'optimize_samples': optimize_samples,
            'reweight_sample_scale' : 30
        }

        X,Y_labels,cluster_means,cluster_covariances,cov_type,samples,plot_mean_meas,msgs = find_clusters(**clustering_config)

        fp_counts = np.unique(Y_labels[Y_labels>-1],return_counts=True)[1]

        # convert to workspace dims'
        if cluster_by_plot_idx: 
            fp_means = np.hstack([ws_conversion(vals[:,None],self.robot_lim[[idx]],self.tray_lim[[idx]]) for vals,idx in 
            zip(cluster_means.T,plot_idx)])
        else: 
            fp_means = ws_conversion(cluster_means,self.robot_lim,self.tray_lim)

        cprint('[TEST] plotting','cyan')
        ## plot
        colors = ['g','b','m','c','y','r']
        fig,axs = plt.subplots(1,2,figsize=(8,4))
        plot_gmm_results(X, Y_labels, cluster_means, cluster_covariances, axs[0],color_iter=colors,plot_idx=plot_idx)
        ax = axs[1]
        heatmap = ax.tricontourf(*samples[:,plot_idx].T,plot_mean_meas,cmap='gist_heat',levels=10)
        for c in heatmap.collections:
            c.set_edgecolor("face")
            c.set_rasterized(True)
        ax.set_title('Fingerprint Locations')
        [ax.scatter(*mu,color=color,edgecolor='white',marker='s',s=200,label=idx,clip_on=False) for idx,(color,mu) in enumerate(zip(colors,cluster_means[:,plot_idx]))]
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.9))
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.set_xlim(self.robot_lim[plot_idx[0]]*1.15)
            ax.set_ylim(self.robot_lim[plot_idx[1]]*1.15)
            ax.xaxis.set_ticks(np.linspace(*self.robot_lim[plot_idx[0]],5))
            ax.yaxis.set_ticks(np.linspace(*self.robot_lim[plot_idx[1]],5))
            ax.set_xlabel(self.plot_states[0])
            ax.set_ylabel(self.plot_states[1])
        fig.tight_layout()

        msg = 'id\tcolor\tcount\ttray loc'
        msgs.append(msg)
        print(msg)
        for idx,(count,color,loc) in enumerate(zip(fp_counts,colors,fp_means)):
            msg = f'{idx}\t{color}\t{count}\t{np.round(loc,2)}'
            msgs.append(msg)
            print(msg)

        if save_name is not None:
            self.save_name = save_name
            fig.savefig(save_name+'.svg')
            with open(save_name + ".txt","a") as f:
                for msg in msgs:
                    f.write(msg+'\n')
            clustering_config['model'] = str(clustering_config['model']).split('\n') 
            clustering_config['batch'] = self.batch_size
            with open(save_name+'_cluster_config.yaml', 'w') as outfile:
                yaml.dump(clustering_config, outfile, default_flow_style=False)

        if self.render_figs:
            plt.show(block=False)
        figs.append(fig)
        return fp_means, figs


def collect_centers(dir_path,buffer,model_path,model=None):
    from dist_modules.sensor_test_module import SensorTest
    from plotting.plotting_matplotlib import EvalPlotter

    fp = FingerprintBuilder(dir_path=dir_path,model_path=model_path,model=model,buffer=buffer)

    test = SensorTest(None,num_steps=100,init_vel=False)

    ## save start_pos
    pos = ws_conversion(test.xinit,test.robot_lim,test.tray_lim)

    ## get fingerprints
    base_fp_name='explr'
    fp_name = dir_path+f'{base_fp_name}_locs'
    cluster_by_plot_idx = False
    fp_locs,fp_figs = fp.find_clusters(None,fp_name,cluster_by_plot_idx=cluster_by_plot_idx,get_blank=False)


    robot_x = []
    robot_y = []
    robot_force = []

    ## visualize fingerprints
    for fp_id,center in enumerate(fp_locs):
        fingerprint_name = f'{base_fp_name}{fp_id}'

        if not (cluster_by_plot_idx):
            fp_pos = center.copy()
        else:
            # reshape xy to full state
            fp_pos = pos.copy()
            fp_pos[test.plot_idx[0]] = center[0]
            fp_pos[test.plot_idx[1]] = center[1]

        # go to start location
        test.use_pose() # switch back to pose controller
        at_center = test.check_goal_pos(fp_pos,-1)
        if not at_center: 
            cprint("didn't make it to the starting pose","cyan")

        iter_step = 0
        while iter_step < 1:
            if test.got_state and test.got_img and not(test.pause):
                success,out = test.step(0,pos=fp_pos)
                if success:
                    iter_step += 1

        # view location
        x,y,force = out

        # plot
        fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.imshow(fp.reshape(y))
        fig.tight_layout()
        fig.savefig(dir_path+f'center_{fingerprint_name}.svg')
        plt.close(fig)

        robot_x.append(x)
        robot_y.append(y)
        robot_force.append(force)
        
    fingerprint_dict = {
        "fp_locs":fp_locs,
        "robot_x":robot_x,
        "robot_y":robot_y,
        "robot_force":robot_force
    }
    pickle.dump( fingerprint_dict, open( dir_path+"cluster_locs.pickle", "wb" ) )

    for tmp_plot in fp_figs:
        plt.close(tmp_plot)



def get_centers_no_buff(dir_path,model_path='model_postexplr.pth',buffer='blah', buffer_name='explr_update_info.pickle'):

    fp = FingerprintBuilder(dir_path=dir_path,model_path=model_path,model=None,buffer=buffer,buffer_name=buffer_name)
    if buffer is None: 
        fp.batch_size = 10
    else:
        fp.batch_size = 0

    ## get fingerprints
    base_fp_name='explr'
    fp_name = dir_path+f'{base_fp_name}_locs2'
    cluster_by_plot_idx = False
    fp_locs,fp_figs = fp.find_clusters(None,fp_name,cluster_by_plot_idx=cluster_by_plot_idx,get_blank=False)
    
    fingerprint_dict = {
        "fp_locs":fp_locs,
    }
    pickle.dump( fingerprint_dict, open( dir_path+"cluster_locs.pickle", "wb" ) )

    return fp

import itertools
import matplotlib.pyplot as plt
def get_dists(dir_path,model_path='model_postexplr.pth',buffer='blah', buffer_name='explr_update_info.pickle'):
    fp = FingerprintBuilder(dir_path=dir_path,model_path=model_path,model=None,buffer=buffer,buffer_name=buffer_name)
    fp.model.init[0] = True
    plot_idx = [0,1]

    s_list = []
    d_list = []
    name_list = []


    if 'Z' in fp.states:
        z_names = ['_Z0','_Z1','_Zm1']
        z_vals = [0,1,-1]
    else:
        z_names = ['']
        z_vals = [0]
    for name,idx in zip(['_posz','_negz','_allz'],[0,1,[]]):
        for z_name, z_val in zip(z_names,z_vals):
            fp.lims = torch.tensor(fp.robot_lim,dtype=fp.dtype)
            explr_robot_lim_scale = 1.15
            fp.lims += torch.tile(torch.tensor([[-1.,1.]]),(len(fp.lims),1))*(fp.lims[:,[1]]-fp.lims[:,[0]])*(explr_robot_lim_scale-1.)/2.      
            env_sampler =  torch.distributions.Uniform(*fp.lims[:2].T)
            z_idx = fp.states.rfind('z')
            fp.lims[z_idx,idx] = 0.
            print(fp.lims,name,z_name)

            num_samples = 1000
            def get_corners(self,plot_idx):
                corner_samples = torch.tensor(list(itertools.product(*self.lims[[plot_idx]])),dtype=self.dtype)
                return corner_samples

            def get_others(self):
                if 'Z' in self.states:
                    others = torch.stack([torch.linspace(*l,10) for l in self.lims[2:-1]])
                else:
                    others = torch.stack([torch.linspace(*l,10) for l in self.lims[2:]])
                other_samples = torch.tensor(list(itertools.product(*others)),dtype=self.dtype)
                return other_samples
            
            def combine(self,samples,others):
                if 'Z' in self.states:
                    return  torch.vstack([torch.hstack([*d,torch.ones(1)*z_val]) for d in list(itertools.product(samples,others))])
                else: 
                    return  torch.vstack([torch.hstack(d) for d in list(itertools.product(samples,others))])
                    

            corners = get_corners(fp,[0,1])
            samples = env_sampler.sample((num_samples,))
            samples = torch.vstack([samples,corners])
            num_plot_samps = samples.shape[0]
            others =  get_others(fp)
            
            samples = combine(fp,samples,others).to(fp.dtype)

            meas = []
            out = fp.vae_buffer.sample(10)
            if fp.learn_force: 
                vals = zip(out[0].unsqueeze(1),out[1].unsqueeze(1),out[2].unsqueeze(1))
            else:
                vals = zip(out[0].unsqueeze(1),out[1].unsqueeze(1))
            for v in vals: 
                fp.model.update_dist(*v)
                meas.append(fp.model.pdf_torch(samples).numpy())
            entropy_dist = np.vstack(meas).mean(0)
            entropy_dist = renormalize(entropy_dist)
            entropy_dist = entropy_dist.reshape(num_plot_samps,-1).mean(1)
            s_list.append(samples.reshape(num_plot_samps,-1,len(fp.states))[:,0,plot_idx].numpy())
            d_list.append(entropy_dist)
            name_list.append(dir_path+'entropy_dist'+name+z_name+'.pdf')
        
    # _min = np.min(np.stack(d_list))
    # _max = np.max(np.stack(d_list)) 
    for samples,entropy_dist,name in zip(s_list,d_list,name_list):
        fig,axs = plt.subplots(1,1,figsize=(4,4))
        axs.tricontourf(*samples.T, entropy_dist, levels=30,cmap='gist_heat') # ,vmin=_min,vmax=_max)
        axs.set_aspect('equal', 'box')
        axs.set_xlabel('x')
        axs.set_ylabel('y')
        fig.tight_layout()
        fig.savefig(name)
        plt.close(fig)

    return fp
