#!/usr/bin/env python

########## global imports ##########
import numpy as np
import numpy.random as npr
from scipy.spatial.transform import Rotation
from scipy.interpolate import NearestNDInterpolator, RBFInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal as mvn
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import os
import yaml

import torch
from torch.distributions import Normal
from termcolor import cprint
import itertools

from argparse import Namespace
np.set_printoptions(precision=3)
import time
from threading import Thread

########## local imports ##########
import sys, os
from franka.franka_utils import ws_conversion
from plotting.plotting_matplotlib import EvalPlotter,set_mpl_format
from control.klerg_utils import *
from .utils import set_seeds, get_num_cores

set_mpl_format()

def get_pairs(num_items):
    vals = []
    for x in range(num_items):
        for y in range(num_items):
            if not(x==y):
                vals.append(np.array([x,y]))
    vals = np.array(vals)
    return np.unique(np.sort(vals,axis=1),axis=0)

@torch.no_grad()
def get_dist(method,z1_mu,z1_logvar,z2_mu,z2_logvar,as_numpy=True):
    diff = z1_mu - z2_mu
    if (z1_logvar is not None) and (z2_logvar is not None):
        z1_var = z1_logvar.exp()
        z2_var = z2_logvar.exp()
    if "L2" in method:
        # out = torch.linalg.norm(diff,dim=list(range(1,len(diff.shape)))) # broke with pytorch 2.0
        out = torch.sqrt(torch.sum(diff**2,dim=list(range(1,len(diff.shape)))))
    elif 'logprob' in method:
        q = Normal(z1_mu, z1_logvar.exp())
        out = -torch.mean(q.log_prob(z2_mu),dim=1)
    elif "KL" in method:
        mu_diff = torch.sum((z1_var+torch.square(diff))/(2*z2_var),dim=1)
        var_diff = torch.sum((z2_logvar/2-z1_logvar/2),dim=1)
        out = var_diff + mu_diff - 0.5*diff.shape[1]
    elif "BC" in method: # bhattacharyya
        mu_diff = torch.sum(torch.square(diff)/(z1_var+z2_var),dim=1)
        var_prod = torch.sum(torch.log((z1_var+z2_var)/2)-z1_logvar/2-z2_logvar/2,dim=1)
        out = 0.25*mu_diff + 0.5*var_prod
    else:
        raise ValueError(f'requested method {method} not defined')

    if as_numpy:
        return out.detach().cpu().numpy()
    else:
        return out

class FingerprintID(object):
    def __init__(self,target_dist,fingerprint_names,fingerprint_method,num_steps,save_name, model_path='model_final.pth',fingerprint_path='eval/',explr_states='xy',test_path='./data/intensity/entklerg_0000/',dist_method='L2',error=False,render_figs=False):

        # params
        self.test_path = test_path
        self.fingerprint_names = fingerprint_names
        self.fingerprint_method = fingerprint_method
        self.num_fingerprints = len(fingerprint_names)
        self.model_path = model_path

        # path to saved vars
        self.dir_path = test_path
        # save path
        self.fpath = self.dir_path + fingerprint_path+save_name.split('/')[0]
        if os.path.exists(self.fpath) == False:
            os.makedirs(self.fpath)
        self.fpath = self.dir_path + fingerprint_path+save_name

        # load variables
        with open(self.dir_path + "/config.yaml","r") as f:
            params = yaml.load(f, Loader=yaml.FullLoader)
        self.map_dict(params)
        # initalize buffers
        self.render_figs = render_figs
        self.robot_lim = np.array(self.robot_lim)
        self.tray_lim = np.array(self.tray_lim)
        self.dtype = eval(self.dtype)
        self.explr_states = explr_states
        self.format_state_indexing()
        if 'plot_idx' not in params.keys():
            self.plot_idx = [self.states.rfind(s) for s in self.plot_states]
        plot_idx = [self.explr_states.rfind(s) for s in self.plot_states]
        self.corner_samples = np.array(list(itertools.product(*self.robot_lim[plot_idx])))

        # seeds
        set_seeds(self.seed)

        ## Initialize VAE
        loaded_info = torch.load(self.dir_path + model_path)
        if not isinstance(loaded_info,dict):
            ## raw model load
            self.model = loaded_info.to(self.device)
        else:
            ## load state_dict only
            from .utils import build_arg_dicts
            from vae import get_VAE
            VAE = get_VAE(self.learn_force)
            model_dict,_ = build_arg_dicts(self,None)
            self.model = VAE(**model_dict).to(self.device)
            self.model.load_state_dict(loaded_info,strict=False)
        # print(self.model)
        self.model.eval()
        # self.model = torch.jit.script(self.model)
        # self.model = torch.jit.freeze(self.model)

        self.reshape = lambda y: np.clip(y,0,1).T

        self.dist_method = dist_method #  ['L2','KL','BC']
        self.error = error
        self.load_fingerprints(fingerprint_path,fingerprint_method,plot_fp_centers=not(self.error))
        if not(self.error):
            self.get_separation(methods = [self.dist_method])

        self.keep_angles=False
        self.reflect_w=True

        # chunk model
        num_threads = get_num_cores()
        possible_chunks=np.arange(1,num_threads+1)
        chunks=int(possible_chunks[(self.num_fp_samples % possible_chunks) == 0][-1] )
        self.model.build_chunk_decoder(chunks)

        # specify target dists
        if self.error:
            self.target_dists = [target_dist(explr_states=self.explr_states,plot_idx=plot_idx,
                                    lims = self.robot_lim,
                                    clip=1e-5,thresh=np.sqrt(np.product(self.image_dim)), # high = low error  (more likely)
                                    name=[name,self.dist_method,'OutputError'],
                                    center=self.fingerprint_dicts[idx]["center"],
                                    center_img=self.fingerprint_dicts[idx]["center_img"])
                                    for idx,name in enumerate(self.fingerprint_names)]
        else:
            self.target_dists = [target_dist(explr_states=self.explr_states, plot_idx=plot_idx, lims = self.robot_lim,
                                    thresh=self.distance_thresh[self.dist_method].min,
                                    # thresh=self.distance_thresh[self.dist_method+'/'+name].mean,
                                    clip=self.distance_thresh[self.dist_method].max*2,
                                    name=[name,self.dist_method,'LatentSpace'],
                                    center=self.fingerprint_dicts[idx]["center"],
                                    center_img=self.fingerprint_dicts[idx]["center_img"])
                                    for idx,name in enumerate(self.fingerprint_names)]
        self.tdist_threads = [None]*self.num_fingerprints
        # debug plots (img)
        self.fingerprint_path=fingerprint_path
        self.guess = None

    def map_dict(self, user_info):
        for k, v in user_info.items():
            setattr(self, k, v)

    def format_state_indexing(self):
        state_dict = {'x': 0, 'y': 1, 'z': 2, 'r': 3, 'p': 4, 'w': 5}
        self.xyzrot_to_state = []
        self.explr_idx = []
        for state in self.explr_states:
            # get locations of explr_idx given states
            self.explr_idx.append(self.states.rfind(state))
            # get location of states given default order
            self.xyzrot_to_state.append(state_dict[state])

        assert not((np.array(self.explr_idx)==-1).any()),'requested exploration state not present in states list'

        if 'w' in self.states:
            self.w_idx = [self.states.rfind('w')]
            self.xyz_idx = [self.states.rfind("x"),self.states.rfind('y'),self.states.rfind('z')]

    def load_fingerprints(self,fingerprint_path,fingerprint_method,plot_fp_centers=True,downsample=1):
        self.fingerprint_dicts = []
        self.fingerprint_dicts_torch = []
        self.num_fp_samples = 0

        for idx,fingerprint_name in enumerate(self.fingerprint_names):
            # load saved fingerprints
            with open(self.dir_path+f"{fingerprint_path}/{fingerprint_name}_{fingerprint_method}.pickle","rb") as f:
                fingerprint_dict = pickle.load(f)
            self.fingerprint_dicts.append(fingerprint_dict)
            # torchify
            fp_dict_torch = {}
            for key in self.fingerprint_dicts[idx].keys():
                if downsample > 1 and ("center" not in key):
                    self.fingerprint_dicts[idx][key] = self.fingerprint_dicts[idx][key][::downsample]
                fp_dict_torch[key] = self.format_tensor(self.fingerprint_dicts[idx][key])
            self.fingerprint_dicts_torch.append(fp_dict_torch)
            self.num_fp_samples = max(self.num_fp_samples,self.fingerprint_dicts[idx]['x'].shape[0])

        # optional plotting
        if plot_fp_centers:
            from matplotlib.gridspec import GridSpec
            num_fp = len(self.fingerprint_names)
            fig = plt.figure(figsize=(4*1.25,2.5*max(1,(num_fp/2.5))))
            gs = GridSpec(num_fp,2, figure=fig)
            ax = fig.add_subplot(gs[:,0]) # span
            axs = [fig.add_subplot(gs[idx, 1]) for idx in range(num_fp)]
            ax.set_title(self.model_path.replace("/","\n"))
            for idx,(fingerprint_name,ax2) in enumerate(zip(self.fingerprint_names,axs)):
                ax.scatter(*self.fingerprint_dicts[idx]["center"][:2],edgecolor='white',marker='s',s=200,label=fingerprint_name)
                # print(fingerprint_name,self.fingerprint_dicts[idx]["center"])
                ax2.imshow(self.fingerprint_dicts[idx]["center_img"])
                ax2.set_aspect('equal', 'box')
                ax2.set_ylabel(fingerprint_name)
                ax2.xaxis.set_ticks([])
                ax2.yaxis.set_ticks([])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax.set_aspect('equal', 'box')
            plot_lim  = self.robot_lim[self.plot_idx]
            ax.set_xlim(plot_lim[0])
            ax.set_ylim(plot_lim[1])
            ax.xaxis.set_ticks(np.linspace(*plot_lim[0],5))
            ax.yaxis.set_ticks(np.linspace(*plot_lim[1],5))
            ax.set_xlabel(self.plot_states[0])
            ax.set_ylabel(self.plot_states[1])
            fig.tight_layout()
            fig.savefig(self.fpath+'fp_locs.svg')
            if self.render_figs:
                plt.show(block=False)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                fig.canvas.start_event_loop(1)


    @torch.no_grad()
    def get_separation(self,methods = ['L2','KL','BC']):
        self.distance_thresh = {}
        # get min separation between fingerprints
        for method in methods:
            dists_tmp = []
            for x,y in get_pairs(self.num_fingerprints):
                for mu, var in zip(self.fingerprint_dicts_torch[x]["z_mu"].clone(),
                                    self.fingerprint_dicts_torch[x]["z_var"].clone()):
                    for z_mu, z_var in zip(self.fingerprint_dicts_torch[y]["z_mu"].clone(),
                                          self.fingerprint_dicts_torch[y]["z_var"].clone()):
                        dists_tmp.append([mu,var,z_mu,z_var])
            dists_tmp = get_dist(method,*map(torch.stack,zip(*dists_tmp)))
            if not(method in self.distance_thresh):
                self.distance_thresh[method] = Namespace()
            self.distance_thresh[method].min = np.min(dists_tmp)
            self.distance_thresh[method].max = np.max(dists_tmp)
            self.distance_thresh[method].mean = np.mean(dists_tmp)
            cprint(f'{method} (all) distance_thresh: min = {self.distance_thresh[method].min:.3f} max = {self.distance_thresh[method].max:.3f}','magenta')

        # get separation within fingerprints
        for idx,fingerprint_name in enumerate(self.fingerprint_names):
            fp_dict_torch = self.fingerprint_dicts_torch[idx]
            for method in methods:
                dists_tmp = []
                for x, (mu, var) in enumerate(zip(fp_dict_torch["z_mu"].clone(),
                                                    fp_dict_torch["z_var"].clone())):
                    for y, (z_mu, z_var) in enumerate(zip(fp_dict_torch["z_mu"].clone(),
                                                            fp_dict_torch["z_var"].clone())):
                        if not (x == y):
                            dists_tmp.append([mu,var,z_mu,z_var])
                dists_tmp = get_dist(method,*map(torch.stack,zip(*dists_tmp)))

                thresh_name = method + '/' + fingerprint_name
                if not(thresh_name in self.distance_thresh):
                    self.distance_thresh[thresh_name] = Namespace()
                self.distance_thresh[thresh_name].min = np.min(dists_tmp)
                self.distance_thresh[thresh_name].max = np.max(dists_tmp)
                self.distance_thresh[thresh_name].mean = np.mean(dists_tmp)
                cprint(f'{thresh_name} thresh | '+
                f'{self.distance_thresh[thresh_name].min:.3E} | '+
                f'{self.distance_thresh[thresh_name].mean:.3f} | '+
                f'{self.distance_thresh[thresh_name].max:.3f} | ','magenta')

    @torch.no_grad()
    def test_fingerprints(self,test_x,test_y,force=None,update_debug_plots=False,update_prior=True,sync=False):
        # start = time.time()
        if isinstance(test_x,torch.Tensor):
            test_x = test_x.cpu().numpy()
        else:
            test_x = np.array(test_x)
        y = self.format_tensor(test_y)
        # loop through fingerprints        
        old_threads = self.tdist_threads.copy()
        self.tdist_threads = [self.process_fingerprint(test_x,y,idx,update_prior=update_prior,tdist_thread=fut) for idx,fut in enumerate(old_threads)]
        if sync and (self.tdist_threads[0] is not None):
            for fut in self.tdist_threads: 
                fut.join()

    @torch.no_grad()
    def process_fingerprint(self,test_x,test_y,fp_idx,update_prior=True,plot_belief=False,use_thread=False,tdist_thread=None):
        # pull saved vars
        mu_stored  = self.fingerprint_dicts_torch[fp_idx]["z_mu"].clone()
        logvar_stored  = self.fingerprint_dicts_torch[fp_idx]["z_var"].clone()

        # test y at stored seed locations
        seed_x  = self.fingerprint_dicts_torch[fp_idx]["x"].clone()
        seed_y = test_y.repeat(seed_x.shape[0],*[1]*len(test_y.shape))

        out = self.model(seed_x, seed_y)
        img_pred, z_mu, z_logvar = out[0],out[2],out[3]

        if self.error:
            dists = get_dist('L2',img_pred,None,seed_y,None)
        else:
            dists = get_dist(self.dist_method,mu_stored,logvar_stored,z_mu,z_logvar)

        self.push_update(test_x.copy(),dists,fp_idx)

        if update_prior: 
            if not use_thread:
                self.target_dists[fp_idx].update_prior()
            elif (tdist_thread is None) or not(tdist_thread.is_alive()):
                tdist_thread = Thread(target=self.target_dists[fp_idx].update_prior)
                tdist_thread.start()



        if plot_belief and (fp_idx==2) and ('L2' in self.dist_method):
            plot_idx = np.argmin(dists)
            # if plot_idx < len(img_pred):
            y_pred = img_pred[plot_idx].detach().cpu().numpy().squeeze()
            latent_space =  [z_mu[plot_idx].detach().cpu().numpy().squeeze(),
                            z_logvar[plot_idx].exp().detach().cpu().numpy().squeeze()]
            self.show_belief_fig(test_y.detach().cpu().numpy().squeeze(),y_pred,latent_space)
            # else:
            #     plot_idx-=len(img_pred)
            #     y_pred = img_pred_flip[plot_idx].detach().cpu().numpy().squeeze()
            #     latent_space =  [z_mu_flip[plot_idx].detach().cpu().numpy().squeeze(),
            #                     z_logvar_flip[plot_idx].exp().detach().cpu().numpy().squeeze()]
            #     self.show_belief_fig(flip_y.detach().cpu().numpy().squeeze(),y_pred,latent_space)

        return tdist_thread

    def show_belief_fig(self,y,y_pred,latent_space):
        if self.guess is None:
            self.guess = EvalPlotter(path=self.dir_path,sim=self.pybullet,method=self.explr_method,save_folder='',render=True,plot_seed=False,plot_z=True)

        self.guess.update(None,self.reshape(y).copy(),self.reshape(y_pred).copy(),latent_space)

    def push_update(self,test_state,vals,fp_idx):
        fingerprint_states = self.fingerprint_dicts[fp_idx]["x"].copy()
        mean_fingerprint_state = self.fingerprint_dicts[fp_idx]["center"].copy()

        orig_len = fingerprint_states.shape[0]

        # find most likely fingerprint(s)
        subset = np.argsort(vals)[:1]
        vals = vals[subset]
        fingerprint_states = fingerprint_states[subset]

        if self.error:
            self.target_dists[fp_idx].push_batch(test_state[self.explr_idx],vals)
        else:
            if 'w' in self.states:
                # convert angles from tray space
                fingerprint_states[:,self.w_idx] = ws_conversion(fingerprint_states[:,self.w_idx],self.robot_lim[self.w_idx],self.tray_lim[self.w_idx])
                test_state[self.w_idx] = ws_conversion(test_state[self.w_idx],self.robot_lim[self.w_idx],self.tray_lim[self.w_idx])
                mean_fingerprint_state[self.w_idx] = ws_conversion(mean_fingerprint_state[self.w_idx],self.robot_lim[self.w_idx],self.tray_lim[self.w_idx])

                # then get rotation
                # rot  = Rotation.from_euler('z',test_state[self.w_idx]-fingerprint_states[:,self.w_idx]).as_matrix() # or fp_rot.T @ test_rot
                fp_rot  = Rotation.from_euler('z',fingerprint_states[:,self.w_idx]).as_matrix()
                fp_rotT = np.transpose(fp_rot,axes=(0,2,1))
                test_rot  = Rotation.from_euler('z',test_state[self.w_idx]).as_matrix()
                mean_rot  = Rotation.from_euler('z',mean_fingerprint_state[self.w_idx]).as_matrix()

                diff = mean_fingerprint_state[self.xyz_idx]-fingerprint_states[:,self.xyz_idx]
                test_xyz = test_state[self.xyz_idx].copy()
                if 'z' not in self.states:
                    diff[:,-1] = 0.
                    test_xyz[-1] = 0.
                belief_xyz = test_xyz  + (( fp_rotT @ test_rot) @ diff[:,:,None]).squeeze(-1)

                # belief_w = test_state[self.w_idx] + (-fingerprint_states[:,self.w_idx] + mean_fingerprint_state[self.w_idx]) # or fp_rot.T @ mean_rot @ test_rot
                belief_rot = Rotation.from_matrix(fp_rotT @ mean_rot @ test_rot).as_euler('xyz')
                belief_rot[:,0] = belief_rot[:,0] % (2 * np.pi) # wrap btwn 0 and 2*pi
                belief_rot[:,1:] = (belief_rot[:,1:] + np.pi) % (2 * np.pi) - np.pi # wrap btwn -pi and pi

                if self.reflect_w:
                    orig_len = belief_rot.shape[0]
                    belief_xyz = belief_xyz.repeat(2,0)
                    belief_rot = belief_rot.repeat(2,0)
                    vals = vals.repeat(2,0)
                    belief_rot[orig_len:,2] +=  2*np.pi*np.sign(belief_rot[orig_len:,2]) # reflect w

                if not self.keep_angles:
                    # convert angle back to tray space
                    belief_rot[:,2] = ws_conversion(belief_rot[:,[2]],self.tray_lim[self.w_idx],self.robot_lim[self.w_idx]).squeeze()
                    test_state[self.w_idx] = ws_conversion(test_state[self.w_idx],self.tray_lim[self.w_idx],self.robot_lim[self.w_idx],)
                # extract current state indexes
                belief_fingerprint_state = np.hstack([belief_xyz,belief_rot])[:,self.xyzrot_to_state]
            else:
                belief_fingerprint_state = test_state[self.explr_idx]-fingerprint_states[:,self.explr_idx]+mean_fingerprint_state[self.explr_idx]

            self.target_dists[fp_idx].push_batch(belief_fingerprint_state,vals)

    def format_tensor(self,x):
        if not isinstance(x,torch.Tensor):
            return torch.as_tensor(x.copy()).to(device=self.device,dtype=self.dtype)
        else:
            return x

def rescale(x,old_minmax,new_minmax):
    return (x - old_minmax[0])/(old_minmax[1]-old_minmax[0])*(new_minmax[1]-new_minmax[0]) + new_minmax[0]

def meas_footprint_vec(locs, samples, explr_idx, std, separate=False, invert=False, get_both=False):
    """ footprint of measurement"""
    std = np.clip(std,1e-6,None)
    inner = np.square(locs[None,:,explr_idx]-samples[:,None,:])/np.abs(std)
    pdf = -0.5 * np.sum(inner, -1)
    pdf = np.exp(pdf)
    # pdf = np.mean(pdf, -1)
    return pdf

class FingerprintDist(object):
    def __init__(self,explr_states='xy',plot_idx=[0,1],capacity=50000,scale=None,thresh=None, clip=None, lims=[[-1,1]]*2,name=None,center=None,center_img=None): # scale was 0.01
        self.name = name
        self.explr_states = explr_states
        self.update_idx = np.arange(len(self.explr_states))
        # self.update_idx = np.array([self.explr_states.rfind("x"),self.explr_states.rfind('y'),self.explr_states.rfind('w')])
        # self.update_idx = np.delete(self.update_idx,self.update_idx==-1)
        # self.z_idx = self.explr_states.rfind('z')
        self.plot_idx = plot_idx
        self.capacity = capacity
        self.scale = scale
        self.thresh = thresh
        self.clip = clip
        self.lims = np.array(lims)
        self.center = center
        self.center_img = center_img

        self.full_buffer = False
        self.position = 0
        self.count=0
        self.env_path = np.empty([self.capacity,len(self.explr_states)])
        self.env_path_val = np.empty(self.capacity)

        self.init = False
        self.invert = False

        self.prior = None
        self.prior_var = None
        self.build_grid()

    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        val = np.ones(x.shape[0])*0.5
        # val /= np.sum(val)
        # val += 1e-5
        return val

    def init_normal_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        val = mvn.pdf(x, mean=np.zeros((len(x.shape),1)), cov=0.5)
        val /= np.sum(val)
        val += 1e-5
        return val

    def process_meas(self,x):
        if self.thresh is None:
            return x
        tmp = self.thresh - x
        tmp[tmp>0] /= self.thresh
        tmp[tmp<0] /= (self.clip - self.thresh )
        # tmp = np.clip(tmp,-1.,1.)
        tmp = np.tanh(tmp)
        return tmp #+1

    def get_meas(self,separate=False):
        if (self.position > 0) or (self.full_buffer):
            if self.full_buffer:
                locs = self.env_path.copy()
                vals = self.env_path_val.copy()
            else:
                locs = self.env_path[:self.position].copy()
                vals = self.env_path_val[:self.position].copy()
            vals = self.process_meas(vals)
        else:
            raise ValueError("need measurements to format")

        if separate:
            return locs,vals
        else:
            return zip(locs,vals)

    def format_meas(self,scale):
        locs,val = self.get_meas(separate=True)
        args = {}
        args['scale'] = scale
        args['locs'] = locs
        args['std'] = val
        return args

    def build_grid(self):
        self.extra_idx = tuple(tuple([x for x in np.arange(len(self.update_idx)) if x not in self.plot_idx]))
        if 'w' in self.explr_states:
            w_idx = self.explr_states.rfind('w')
            self.lims[w_idx] *= 1.33 # fill in remaining angles
        self.lims *= 1.15 # explr_robot_lim_scale
        num_samples = 50
        self.xy_mesh = np.meshgrid(*np.linspace(*self.lims[self.plot_idx].T,num_samples).T)
        self.xy_grid = np.c_[[m.ravel() for m in self.xy_mesh]].T
        mesh_spacing = np.linspace(*self.lims[self.update_idx].T,num_samples)
        self.mesh = np.meshgrid(*mesh_spacing.T)
        self.grid = np.c_[[m.ravel() for m in self.mesh]].T
        self.num_samples = [num_samples]*len(self.update_idx)
        if self.scale is None:
            self.scale = np.max(mesh_spacing[1]-mesh_spacing[0])*2.5 #  # 1.5
            # print(self.scale)


    def save_results(self,fpath,iter_step):
        fingerprint_dict = {
            "name": self.name,
            "prior": self.prior,
            "prior_var": self.prior_var,
            "lims": self.lims,
            "plot_idx": self.plot_idx,
            "extra_idx": self.extra_idx,
            "num_samples": self.num_samples,
            "center":self.center,
            "center_img":self.center_img,
            "scale":self.scale,
            "states":self.explr_states
        }
        pickle.dump( fingerprint_dict, open( fpath + f"_belief_{self.name[0]}_{self.name[1]}_{self.name[2]}_{iter_step}.pickle", "wb" ) )

    def update_prior(self,debug_plots=False,smooth=False):
        if self.prior is None:
            prior = self.init_uniform_grid(self.grid)
            self.prior = prior.copy()
            self.prior_var = np.ones(prior.shape)*2.

        loc,val = self.get_meas(separate=True)
        if len(loc.shape) < 2:
            loc = np.expand_dims(loc,0)
        n = loc.shape[0]
        ## measurement location
        meas_map = meas_footprint_vec(samples=self.grid,explr_idx=self.update_idx,locs=loc,std=self.scale/2.)
        meas_map = renormalize(meas_map,0)
        
        use_mask = False

        ## measurement vals
        meas = np.ones((*self.prior.shape,n))*val
        # meas = meas_map.copy()*val
        meas = (meas/2)+0.5
        meas_var = np.mean(meas_map,1)
        meas_var = renormalize(meas_var)
        meas_var = rescale(meas_var,[0.,1.],[50.*self.scale,self.scale]) # invert distribution and rescale max
        # meas_var = np.ones(self.prior.shape)*self.scale

        ## normal distribution 
        posterior_var = 1./(1./self.prior_var+n/meas_var) #
        posterior = posterior_var*(self.prior/self.prior_var+ np.sum(meas,1)/meas_var)
        if use_mask:
            meas_mask = meas_map > meas_map.min(0)
            meas_mask = np.bitwise_or.reduce(meas_mask,axis=1)
            ## copy over previous
            not_mask = np.bitwise_not(meas_mask)
            posterior_var[not_mask] = self.prior_var.copy()[not_mask]
            posterior[not_mask] = self.prior.copy()[not_mask]
        # print(posterior.min(),posterior.max(),posterior_var.min(),posterior_var.max())

        # smooth periodically
        if smooth and self.count > 0 and np.any(np.arange(self.count,self.count+n) % 100 == 0):
            tmp_dist = posterior.reshape(self.num_samples)
            tmp_dist = rescale(tmp_dist,[tmp_dist.min(),tmp_dist.max()],[tmp_dist.max(),tmp_dist.min()]) # flip
            tmp_dist = gaussian_filter(tmp_dist,sigma=1,mode='nearest') # smooth low first
            tmp_dist = rescale(tmp_dist,[tmp_dist.min(),tmp_dist.max()],[tmp_dist.max(),tmp_dist.min()]) # flip back
            tmp_dist = gaussian_filter(tmp_dist,sigma=1,mode='nearest') # then smooth high
            posterior = tmp_dist.flatten()

        self.count += n 
        self.prior = posterior.copy()
        self.prior_var = posterior_var.copy()

        self.clear_batch()

    def pdf(self,samples,override_invert=False,plot=False,use_grid=False):
        if use_grid: # overwrite input samples
            samples = self.grid
        if (self.init and self.prior is not None):
            if use_grid:
                dist = self.prior.copy()
            else:
                # interp = NearestNDInterpolator(self.grid, self.prior)
                interp = RBFInterpolator(self.grid, self.prior, kernel='linear')# , smoothing=.01)
                dist = interp(samples)
            if self.invert and not override_invert:
                dist = -dist+np.max(dist)+np.min(dist) # invert distribution and shift min to 0
            return dist
        else:
            vals = self.init_uniform_grid(samples)
            return vals

    def push(self,state,val):
        if (not self.full_buffer) and ((self.position + 1 ) == self.capacity):
            self.full_buffer = True
        self.env_path[self.position] = state
        self.env_path_val[self.position] = val
        self.position = (self.position + 1) % self.capacity

    def push_batch(self,state,val):
        num_items=val.shape[0]
        if (not self.full_buffer) and ((self.position + num_items ) >= self.capacity):
            self.full_buffer = True
            # print('buffer full', self.position)
        self.env_path[self.position:self.position+num_items] = state
        self.env_path_val[self.position:self.position+num_items] = val
        self.position = (self.position + num_items) % self.capacity

    def clear_batch(self):
        self.full_buffer = False
        self.position = 0
        self.env_path = np.empty([self.capacity,len(self.explr_states)])
        self.env_path_val = np.empty(self.capacity)

def process_grid_dist(p_list,target_dists,angle_method):
    p_processed = []
    if angle_method in ['mean','range','max']:
        for p,target_dist in zip(p_list,target_dists):
            p =  p.reshape(target_dist.num_samples)
            p_min = p.min(target_dist.extra_idx)
            p_max = p.max(target_dist.extra_idx)
            p_mean = p.mean(target_dist.extra_idx)
            if angle_method == 'mean' :
                p_processed.append(p_mean)
            elif angle_method == 'range' :
                p_processed.append(p_max-p_min)
            elif angle_method == 'max' :
                p_processed.append(p_max)
    elif angle_method in ['maxNorm', 'minNorm', 'WeightedAvg1', 'WeightedAvg2']: # normalize across fingerprints
        mask = False
        masks = []
        p_reshaped = []
        ## 1) find the object (min values @ averaged angle)
        for p,target_dist in zip(p_list,target_dists):
            p =  p.reshape(target_dist.num_samples)
            if mask:
                p_mean = p.mean(target_dist.extra_idx)
                thresh = np.quantile(p_mean,0.15)
                p_objs = p_mean < thresh
                masks.append(p_objs)
            p_reshaped.append(p)
        ## 2) process angles
        p_reshaped = np.array(p_reshaped)
        for idx,target_dist in enumerate(target_dists):
            p_weighted = p_reshaped.copy()
            #### find max value for this index
            if 'max' in angle_method:
                for extra_axis in target_dist.extra_idx:
                    p_tmp = p_weighted[idx].copy()
                    filtered = np.argmax(p_tmp, axis=extra_axis, keepdims=True)
                    p_weighted = np.take_along_axis(p_weighted, np.expand_dims(filtered,axis=[0]), axis=extra_axis+1) # apply filter
            elif 'min' in angle_method :
                for extra_axis in target_dist.extra_idx:
                    p_tmp = p_weighted[idx].copy()
                    filtered = np.argmin(p_tmp, axis=extra_axis, keepdims=True)
                    p_weighted = np.take_along_axis(p_weighted, np.expand_dims(filtered,axis=[0]), axis=extra_axis+1) # apply filter
            elif 'WeightedAvg' in angle_method:
                #### or sort angles min to max for this index
                for extra_axis in target_dist.extra_idx:
                    p_tmp = p_weighted[idx].copy()
                    filtered = np.argsort(p_tmp, axis=extra_axis) # [:,:,-10:]
                    p_sort = np.take_along_axis(p_weighted,np.expand_dims(filtered, axis=[0]),axis=extra_axis+1) # apply filter
                    if '1' in angle_method:
                        p_weighted = np.average(p_sort,weights=0.95**np.arange(p_sort.shape[extra_axis+1],0,-1),axis=extra_axis+1,keepdims=True) # weighted mean
                    else:
                        p_weighted = np.average(p_sort,weights=0.95**np.arange(p_sort.shape[extra_axis+1]),axis=extra_axis+1,keepdims=True) # weighted mean
            p_weighted = p_weighted.squeeze()
            # average across max angles
            p_weighted /= p_weighted.sum(0,keepdims=True)
            if not mask:
                #### filter by object only
                p_processed.append(p_weighted[idx])
            else:
                #### or filter by object and apply mask
                p_objs = masks[idx]
                test_p = p_weighted[idx][p_objs]
                masked_val = test_p.min()

                p_masked = np.ones(p_weighted.shape[1:])*masked_val
                p_masked[p_objs] = test_p

                # p_masked = renormalize(p_masked)
                p_processed.append(p_masked)
    else:
        raise ValueError('invalid method requested')
    return p_processed

class FingerprintsPlotter(object):
    def __init__(self,fingerprint_names,target_dists,lim,corner_samples=None,render=True,window_name=None,rank=None,fig_no=0,no_figs=2,explr_dim=2,vertical=True):
        self.fingerprint_names = fingerprint_names
        self.lims = lim
        # print(self.lims)
        self.render = render
        self.explr_dim = explr_dim
        self.corner_samples = corner_samples
        self.tri = None
        self.cbar = None
        self.cmap = 'gist_heat'
        self.target_dists = target_dists
        self.vertical = vertical
        if vertical:
            self.fig,self.heatmap = plt.subplots(len(fingerprint_names),1,figsize=(2.5,2.3*len(fingerprint_names)))
            for ax,fingerprint_name in zip(self.heatmap,fingerprint_names):
                self.build_axes('',f'Belief Grid {fingerprint_name[-1]}',ax)
        else:
            self.fig,self.heatmap = plt.subplots(1,len(fingerprint_names),figsize=(2.*len(fingerprint_names),2.5),sharey=True)
            for ax,fingerprint_name in zip(self.heatmap,fingerprint_names):
                self.build_axes(f'{fingerprint_name}','',ax)
        if self.target_dists is not None:
            self.update()
            # self.fig.tight_layout()
            ## try to tile windows (if given enough info)
            if rank is not None:
                backend = mpl.get_backend()
                if backend == 'TkAgg' or backend == 'Qt5Agg':
                    if window_name is not None:
                        self.fig.canvas.manager.set_window_title(window_name)
                if backend == 'TkAgg':
                    self.fig.canvas.toolbar.pack_forget() # remove bottom status bar
                    screen_width,screen_height = self.fig.canvas.manager.window.wm_maxsize()
                    window_width=self.fig.canvas.manager.window.winfo_reqwidth()
                    window_height=self.fig.canvas.manager.window.winfo_reqheight()
                    no_tall = np.floor(screen_height/window_height)+1
                    no_wide = np.floor(screen_width/window_width)
                    x = (np.floor((rank-1)*no_figs / no_tall)%no_wide)*window_width
                    y = (((rank-1)*no_figs + fig_no) % no_tall)*window_height
                    self.fig.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
            if self.render:
                plt.ion()
                plt.show(block=False)

    def refresh(self):
        if self.fig.stale:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(1)

    def build_axes(self,title,ylabel,ax):
        ax.set_title(title)
        if (title == '' ):
            ax.xaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticklabels([])
            ax.yaxis.set_ticks([])
        else:
            ax.axes.xaxis.set_ticks(np.linspace(self.lims[0,0],self.lims[0,1],5))
            ax.axes.yaxis.set_ticks(np.linspace(self.lims[1,0],self.lims[1,1],5))
            ax.set_xlim(self.lims[0,0],self.lims[0,1])
            ax.set_ylim(self.lims[1,0],self.lims[1,1])
        ax.set_ylabel(ylabel)
        ax.set_aspect('equal', adjustable='box')



    def process_tdists(self,norm=False,use_grid=True,angle_method='mean'):
        if use_grid:
            samples = None
        else:
            numpts = 5000
            samples = npr.uniform(self.lims[:,0], self.lims[:,1],size=(numpts, self.explr_dim))
            if self.corner_samples is None:
                samples = np.vstack([samples, self.corner_samples])
        p_list = np.array([target_dist.pdf(samples,plot=True,use_grid=use_grid).squeeze() for target_dist in self.target_dists])
        if norm:
            p_list = np.array([renormalize(p) for p in p_list])
        if use_grid:
            p_list_processed = process_grid_dist(p_list,self.target_dists,angle_method)
            smooth = False
            if smooth:
                ## smooth output for plotting
                p_list_processed = [gaussian_filter(p,sigma=1,mode='nearest') for p in p_list_processed]
            samples = [target_dist.xy_mesh for target_dist in self.target_dists]
            p_list = p_list_processed
        else:
            samples = [samples[:,target_dist.plot_idx] for target_dist in self.target_dists]

        return samples,p_list

    def update(self,fname=None,norm=False,use_grid=True,angle_method='mean',p_list=None,samples=None,title=None):
        if fname is not None:
            endings = np.array(['.svg','.png','.pdf'])
            check_endings = [end in fname for end in endings]
            if any(check_endings):
                end = endings[check_endings][0]
            else:
                raise ValueError('invalid fname')

            iter_step = int(fname.split(end)[0].split('_step')[-1])
            if self.vertical:
                self.heatmap[0].set_title(f'belief iteration: {iter_step}')
            else:
                self.fig.suptitle(f'belief iteration: {iter_step}')
        elif title is not None: 
            if self.vertical:
                self.heatmap[0].set_title(title)
            else:
                self.fig.suptitle(title)

        if self.tri is not None:
            if hasattr(self.tri[0],'collections'):
                # removes only the contours, leaves the rest intact
                [c.remove() for cont in self.tri for c in cont.collections]
        if p_list is None:
            samples,p_list = self.process_tdists(norm,use_grid,angle_method)
        _min, _max = np.amin(p_list), np.amax(p_list)
        plotting_args = {'cmap':self.cmap,'levels':30}
        if use_grid:
            tri = [ax.contourf( *mesh,p, **plotting_args) for ax,p,mesh in zip(self.heatmap,p_list,samples)]
        else:
            tri = [ax.tricontourf( *samps.T, p, **plotting_args) for ax,p,samps in zip(self.heatmap,p_list,samples)]
        self.tri = tri
        # for ax,p in zip(self.heatmap,p_list):
        #     ax.set_xlabel(f'min: {p.min():0.4f} max: {p.max():0.4f}')

        if self.vertical:
            locs = self.heatmap
            shift1 = 0.65
            shift2 = 1.4
            width = 0.06
        else:
            locs = self.heatmap[[-1]]
            shift1 = 0.95
            shift2 = 1.05
            width = 0.04

        if self.cbar is None:
            # create color bar
            for ax in self.heatmap:
                box = ax.get_position()
                ax.set_position([box.x0*shift1, box.y0, box.width, box.height])
        else:
            for cbar in self.cbar:
                cbar.remove()
        self.cbar = []
        for ax in locs:
            box = ax.get_position()
            axColor = self.fig.add_axes([box.x0*shift2 + box.width, box.y0+0.15*box.height, width, 0.7*box.height])
            axColor.set_title('High',size='medium')
            axColor.set_xlabel('Low')
            sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=_min, vmax=_max), cmap=self.cmap)
            cbar = self.fig.colorbar(sm, cax = axColor, orientation="vertical")
            cbar.set_ticks([])
            cbar.set_ticklabels([])
            # cbar.set_ticks([_min, _max])
            # cbar.set_ticklabels(['Low', 'High'])
            self.cbar.append(cbar)
        if fname is not None:
            self.save(fname)
        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(1)

    def save(self,fname):
        self.fig.savefig(fname)
