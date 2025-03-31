#!/usr/bin/env python

########## global imports ##########
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import torch
import itertools as it
import yaml 

import time
import copy
from termcolor import cprint

from scipy.cluster.vq import kmeans,kmeans2
from sklearn.cluster import MeanShift
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import seaborn as sns

########## local imports ##########
import sys, os
from .utils import setup, get_num_cores, add_yaml_representers
from plotting.plotting_matplotlib import set_mpl_format
from franka.franka_utils import find_non_vel_locs
from control_torch.klerg_utils import renormalize as renormalize_torch
from control.klerg_utils import *

## change default figure params
set_mpl_format()
add_yaml_representers()

def relabel(Y_labels,new_order,offset=0.):
    label_shifted = Y_labels.copy()
    for old_idx,new_idx in enumerate(new_order):
        label_shifted[Y_labels==old_idx] = new_idx + offset
    return label_shifted

def plot_gmm_results(X, Y_, means, covariances, ax, title='Clusters',color_iter=['r','g','b','k'],plot_idx=[0,1]):
    skip_cov = covariances is None
    if not(skip_cov) and not(means.shape[0] == covariances.shape[0]):
        covariances = covariances[None,:,:].repeat(means.shape[0],0)
    elif skip_cov:
        covariances = [1]*means.shape[0]
    for i, (mean, covar, color) in enumerate(zip(means, covariances, color_iter)):
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        ax.scatter(X[Y_ == i, plot_idx[0]], X[Y_ == i, plot_idx[1]], 3, color=color)
        ax.scatter(*mean[plot_idx], 50, color='k')

        if not(skip_cov) and not(np.all(covar) == 0):
            if covar.size == 1:
                covar = np.eye(2)*covar
            elif len(covar.shape) == 1:
                covar = np.diag(covar[plot_idx])
            else:
                tmp_covar = np.eye(2)
                for idx_x,x in enumerate(plot_idx):
                    for idx_y,y in enumerate(plot_idx):
                        tmp_covar[idx_x,idx_y] = covar[x,y]
                covar = tmp_covar
            v, w = np.linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / np.linalg.norm(w[0])
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean[plot_idx], v[0], v[1], angle=180.0 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)
    ax.set_title(title)

def find_clusters(model,
                  states,plot_idx,tray_lim,robot_lim,robot_ctrl_lim,num_pts,scale=1.25,uniform=True,print_debug=True,
                  num_fingerprints=2,get_blank=True,cluster_by_plot_idx=True,batch=None,
                  sample_method='reweight',cluster_method='mean_shift',optimize_samples=False,
                  cluster_weight=False,reweight_sample_scale=10):

    learn_force = model.learn_force
    if batch is not None: 
        if learn_force:
            test_xs, test_ys, test_forces = batch
        else:
            test_xs, test_ys = batch
    else: 
        test_xs = model.seed_x.detach().clone()
        test_ys = model.seed_y.detach().clone()
        if learn_force: 
            test_forces = model.seed_force.detach().clone()
    batch_size = test_xs.shape[0]

    ## generate samples ##
    if optimize_samples: 
        samples = npr.uniform(*robot_lim.T,size=(num_pts, len(robot_lim)))
        x = torch.as_tensor(samples).to(dtype=model.dtype)

        cluster_by_plot_idx = False
        if print_debug: cprint('[CLUSTERING] optimizing samples','magenta')
        kernel_covar = torch.diag(torch.ones(len(states))*0.001)
        kernel_dist = torch.distributions.MultivariateNormal(torch.zeros(len(states)),kernel_covar)

        from control_torch.barrier import setup_barrier
        non_vel_locs,vel_locs,states = find_non_vel_locs(states)
        barrier,barr_lim = setup_barrier(states,robot_lim,robot_ctrl_lim,non_vel_locs,model.dtype,None,override_use_barrier=True)
        barrier.update_ergodic_dim(len(non_vel_locs))

        def kernel(kernel_dist,x1, x2):
            return kernel_dist.log_prob(x1-x2).exp()

        def kernel_loss(kernel_dist,target_dist,pts): 
            inner_prod = torch.mean(kernel(kernel_dist,pts.unsqueeze(1),pts.unsqueeze(0)))
            return inner_prod - 12 * torch.mean(renormalize_torch(target_dist.pdf_torch(pts))) + torch.mean(barrier(pts))

        target_samples = x.detach().clone()
        optimizer = torch.optim.Adam([target_samples],lr=0.05) 
        target_samples.requires_grad = True

        if get_blank:
            def kernel_loss_neg(kernel_dist,target_dist,pts): 
                inner_prod = torch.mean(kernel(kernel_dist,pts.unsqueeze(1),pts.unsqueeze(0)))
                return -(inner_prod - 2 * torch.mean(renormalize_torch(target_dist.pdf_torch(pts)))) + torch.mean(barrier(pts))
            
            blank_samples = x.detach().clone()
            blank_optimizer = torch.optim.Adam([blank_samples],lr=0.05)
            blank_samples.requires_grad = True 

        start = time.time()
        kernel_opt_iter = 5 # 15
        for i in range(kernel_opt_iter): 
            loss = 0.
            blank_loss = 0.
            if learn_force: 
                vals = zip(test_xs.unsqueeze(1),test_ys.unsqueeze(1),test_forces.unsqueeze(1))
            else: 
                vals = zip(test_xs.unsqueeze(1),test_ys.unsqueeze(1))
            for val in vals:
                model.update_dist(*val)
                loss += kernel_loss(kernel_dist,model,target_samples)
                if get_blank:
                    blank_loss += kernel_loss_neg(kernel_dist,model,blank_samples)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if get_blank:
                blank_optimizer.zero_grad(set_to_none=True)
                blank_loss.backward()
                blank_optimizer.step()
            if print_debug: cprint(f'[CLUSTERING] kernel iter {i+1}/{kernel_opt_iter}, {time.time()-start}','magenta')
        x = target_samples.requires_grad_(False)
        samples = x.numpy()
        if get_blank: 
            blank_samples=blank_samples.requires_grad_(False)
        # print(time.time()-start)
        sampling_msg = f'SAMPLING:\t num figs {batch_size}\t method kernel optimization ({kernel_opt_iter} iters)\t num points {num_pts}\t scale robot lim {scale}\t sampling dim {len(robot_lim)}'
    else:     
        samples = npr.uniform(*robot_lim.T*scale,size=(num_pts, len(robot_lim)))
        x = torch.as_tensor(samples).to(dtype=model.dtype)

        sampling_msg = f'SAMPLING:\t num figs {batch_size}\t method {"uniform"}\t num points {num_pts}\t scale robot lim {scale}\t sampling dim {len(robot_lim)}'

    meas = []
    plot_meas = []

    ## clean up for plotting
    plot_samples = x.detach().clone()
    corner_samples = torch.tensor(list(it.product(*robot_lim[plot_idx]*scale)),dtype=x.dtype)
    tmp_corners = torch.zeros([corner_samples.shape[0],plot_samples.shape[1]],dtype=x.dtype)
    tmp_corners[:,plot_idx] = corner_samples
    plot_samples = torch.vstack([plot_samples, tmp_corners])

    remaining_dim_idx = list(range(len(robot_lim)))
    for idx in np.sort(plot_idx)[::-1]:
        remaining_dim_idx.pop(idx)
    for idx in remaining_dim_idx:
        # plot_samples[:,idx] = np.mean(robot_lim[idx])
        plot_samples[:,idx] = test_xs[:,idx].min().item()

    if learn_force: 
        vals = zip(test_xs.unsqueeze(1),test_ys.unsqueeze(1),test_forces.unsqueeze(1))
    else: 
        vals = zip(test_xs.unsqueeze(1),test_ys.unsqueeze(1))
    for idx,val in enumerate(vals):
        # #### use update dist
        model.update_dist(*val)
        img_var = model.decode_samples_only(x)
        meas.append(img_var.exp().mean(axis=tuple(range(1,len(img_var.shape)))).detach().numpy())
        plot_img_var = model.decode_samples_only(plot_samples)
        plot_meas.append(plot_img_var.exp().mean(axis=tuple(range(1,len(plot_img_var.shape)))).detach().numpy())
        if print_debug: cprint(f'[CLUSTERING] processed fig {idx+1}/{batch_size}','magenta')
    meas = np.stack(meas)
    mean_meas = np.mean(meas,0).squeeze()
    # mean_meas = renormalize(mean_meas)
    mean_meas = mean_meas**3
    # mean_meas = renormalize(mean_meas)
    # print(meas.min(),meas.max(),meas.mean(),np.median(meas))

    plot_meas = np.stack(plot_meas)
    plot_mean_meas = np.mean(plot_meas,0).squeeze()
    plot_mean_meas = renormalize(plot_mean_meas)
    # plot_mean_meas = plot_mean_meas**3

    if print_debug: cprint('[CLUSTERING] reweighting samples','magenta')
    X,Y_labels,cluster_means,cluster_covariances,cov_type,clustering_msg = process_clusters(samples,mean_meas,plot_idx,num_fingerprints,sample_method,cluster_method,optimize_samples,cluster_by_plot_idx,cluster_weight,reweight_sample_scale)

    fp_counts = np.unique(Y_labels[Y_labels>-1],return_counts=True)[1]
    # check for overlap 
    num_fp = len(cluster_means)
    done = False
    while not done and num_fp > 1:
        num_fp = len(cluster_means)
        dists = np.sum((cluster_means[None,:,:] - cluster_means[:,None,:])**2,2) + np.eye(num_fp)
        overlap = dists < 0.04
        if np.any(overlap):
            idx = np.argmax(overlap.sum(1))
            labels = np.arange(num_fp)
            keep_locs = np.delete(labels,idx)
            labels = np.insert(labels,idx,-1)
            
            # relabel
            Y_labels = relabel(Y_labels,labels)
            cluster_means = cluster_means[keep_locs]
            if cluster_covariances is not None and not(cov_type == 'tied'):
                cluster_covariances=cluster_covariances[keep_locs]
        else: 
            done = True

    fp_counts = np.unique(Y_labels[Y_labels>-1],return_counts=True)[1]
    if print_debug: cprint(f'[CLUSTERING] found {fp_counts} clusters','magenta')

    if get_blank:
        if print_debug: cprint('[CLUSTERING] finding blanks','magenta')
        blank_cluster_covariances = None
        if optimize_samples: 
            blank_X = blank_samples[:,plot_idx] # cluster_by_plot_idx
        else: 
            blank_idx = list(set(np.arange(samples.shape[0]))-set(np.unique(idx)))
            tmp_x = samples[blank_idx]
            if cluster_by_plot_idx: 
                blank_X = tmp_x[:,plot_idx]
            else: 
                blank_X = tmp_x
        reweight_msg = f'not obj_idx'

        if 'kmeans' in  cluster_method or ('gmm' in cluster_method):
            # kmeans clustering w/ scipy
            blank_msg = f'BLANK CLUSTERING:\t kmeans\t ' + reweight_msg
            blank_cluster_means,blank_Y_labels = kmeans2(data=blank_X,k=num_fingerprints,minit='points')
        # elif ('gmm' in cluster_method) or ('mixture' in cluster_method):
        #     # gmm clustering w/ sklearn
        #     cov_type = 'tied' # {'full', 'tied', 'diag', 'spherical'}
        #     blank_msg = f'BLANK CLUSTERING:\t GMM, {cov_type}\t ' + reweight_msg
        #     blank_gmm = GaussianMixture(n_components=num_fingerprints,covariance_type=cov_type,n_init=10).fit(X)
        #     blank_cluster_means = blank_gmm.means_
        #     blank_cluster_covariances = blank_gmm.covariances_
        #     blank_Y_labels = blank_gmm.predict(blank_X)
        elif 'shift' in cluster_method:
            blank_msg = f'BLANK CLUSTERING:\t MeanShift\t ' + reweight_msg
            blank_msc = MeanShift(bin_seeding=True,cluster_all=False,min_bin_freq=10)
            blank_msc.fit(blank_X)
            blank_cluster_means = blank_msc.cluster_centers_
            blank_Y_labels = blank_msc.labels_

        ## pad outputs possibly some dimensions are not used for blank clustering
        if optimize_samples: 
            tmp_blank_x = np.zeros([blank_X.shape[0],X.shape[1]])
            tmp_blank_x[:,plot_idx] = blank_X
            blank_X = tmp_blank_x
            tmp_blank_cluster_means = np.zeros([blank_cluster_means.shape[0],cluster_means.shape[1]])
            tmp_blank_cluster_means[:,plot_idx] = blank_cluster_means
            blank_cluster_means = tmp_blank_cluster_means
            if blank_cluster_covariances is not None:
                tmp_blank_cluster_covariances = np.zeros([blank_cluster_covariances.shape[0],cluster_covariances.shape[1]])
                tmp_blank_cluster_covariances[:,plot_idx] = blank_cluster_covariances
                blank_cluster_covariances = tmp_blank_cluster_covariances

        ## sort from largest to smallest
        blank_fp_counts = np.unique(blank_Y_labels[blank_Y_labels>-1],return_counts=True)[1]
        fp_offset = len(cluster_means)
        blank_Y_labels = blank_Y_labels+fp_offset

        if print_debug: cprint(f'[CLUSTERING] found {blank_fp_counts} blanks','magenta')
        ## then combine with object clusters
        X = np.vstack([X,blank_X])
        Y_labels = np.hstack([Y_labels,blank_Y_labels])
        cluster_means = np.vstack([cluster_means,blank_cluster_means])
        if blank_cluster_covariances is not None:
            cluster_covariances = np.vstack([cluster_covariances,blank_cluster_covariances])
        elif cluster_covariances is not None:
            cluster_covariances = np.stack([*[cluster_covariances]*fp_offset,np.diag([0,0,0])])
            cov_type = 'full'

        # check for overlap 
        num_fp = len(cluster_means)
        done = False
        while not done and num_fp > 1:
            num_fp = len(cluster_means)
            if optimize_samples:
                dists = np.sum((cluster_means[None,:,plot_idx] - cluster_means[:,None,plot_idx])**2,2) + np.eye(num_fp)
            else:
                dists = np.sum((cluster_means[None,:,:] - cluster_means[:,None,:])**2,2) + np.eye(num_fp)
            overlap = dists < 0.04
            if np.any(overlap):
                loc = overlap.sum(1)
                idx = np.argwhere(loc == np.max(loc))[-1] # if there are multiple, delete the blank one
                labels = np.arange(num_fp)
                keep_locs = np.delete(labels,idx)
                labels = np.insert(labels,idx,-1)
                
                # relabel
                Y_labels = relabel(Y_labels,labels)
                cluster_means = cluster_means[keep_locs]
                if cluster_covariances is not None and not(cov_type == 'tied'):
                    cluster_covariances=cluster_covariances[keep_locs]
            else: 
                done = True

        fp_counts_combo = np.unique(Y_labels[Y_labels>-1],return_counts=True)[1]
        if print_debug: cprint(f'[CLUSTERING] found {fp_counts_combo} combo','magenta')
    else:
        blank_msg = 'no separate blank clustering'

    msgs = []
    msgs.append(sampling_msg)
    msgs.append(clustering_msg)
    msgs.append(blank_msg)

    return X,Y_labels,cluster_means,cluster_covariances,cov_type,plot_samples.numpy(),plot_mean_meas,msgs

def process_clusters(samples,mean_meas,plot_idx,num_fingerprints,sample_method = 'reweight',cluster_method = 'gmm',optimize_samples=False,cluster_by_plot_idx=True,uniform=True,cluster_weight=False,reweight_sample_scale=10.):

    if 'reweight' in sample_method:
        ## weighted to unweighted
        num_pts = samples.shape[0]
        if optimize_samples:
            num_ds = int(num_pts*reweight_sample_scale)
        else: 
            num_ds = num_pts//2
        idx = np.random.choice(num_pts, p=mean_meas/np.sum(mean_meas),size=num_ds,replace=True)
        # inv_meas = -mean_meas+mean_meas.min()+mean_meas.max()
        # idx_neg = np.random.choice(test_x.shape[0], p=inv_meas/np.sum(inv_meas),size=num_ds,replace=True)
        reweight_msg = f'reweight w/ {num_ds} samples'
    elif 'thresh' in sample_method:
        ## remove values below some threshold
        thresh = 0.5
        idx = np.where(mean_meas > np.quantile(mean_meas,thresh))[0]
        reweight_msg = f'thresh, >{int(thresh*100)}%'
    else:
        idx = np.arange(samples.shape[0])
        reweight_msg = 'no reweight/resampling'
        if not optimize_samples:
            get_blank = False
    if cluster_by_plot_idx: 
        X = np.hstack([samples[idx,plot_idx[0],None],samples[idx,plot_idx[1],None]])
    else: 
        X = samples[idx]
    if cluster_weight: 
        X = np.hstack([samples[idx],mean_meas[idx,None]])
    if not uniform:
        X[:,:-1] += 0.005*np.random.randn(X.shape[0],X.shape[1]-1)

    cluster_covariances = None
    cov_type = None
    if 'kmeans' in cluster_method:
        # kmeans clustering w/ scipy
        clustering_msg = f'CLUSTERING:\t kmeans\t ' + reweight_msg
        cluster_means,Y_labels = kmeans2(data=X,k=num_fingerprints,minit='points')
    elif ('gmm' in cluster_method) or ('mixture' in cluster_method):
        # gmm clustering w/ sklearn 
        cov_type = 'tied' # {'full', 'tied', 'diag', 'spherical'}
        clustering_msg = f'CLUSTERING:\t GMM, {cov_type}\t ' + reweight_msg
        gmm = GaussianMixture(n_components=num_fingerprints,covariance_type=cov_type,n_init=10).fit(X)
        cluster_means = gmm.means_
        cluster_covariances = gmm.covariances_
        Y_labels = gmm.predict(X)
    elif 'shift' in cluster_method:
        clustering_msg = f'CLUSTERING:\t MeanShift\t ' + reweight_msg
        # bandwidth = estimate_bandwidth(X,n_samples=100)/2
        msc = MeanShift(bin_seeding=True,cluster_all=False,min_bin_freq=10) # bandwidth=bandwidth
        msc.fit(X)
        cluster_means = msc.cluster_centers_
        Y_labels = msc.labels_

    if cluster_weight:
        X = X[:,:-1]
        cluster_means = cluster_means[:,:-1]
        if cluster_covariances is not None and not(cov_type == 'tied') and not(cov_type == 'spherical'): 
            cluster_covariances = cluster_covariances[:,:-1,:-1]

    return X,Y_labels,cluster_means,cluster_covariances,cov_type,clustering_msg

class Clustering(object):
    def __init__(self,args):

        # initalize buffers
        self.map_dict(vars(args['args']))
        self.num_pts = 1000 # was 5000
        self.device = 'cpu'

        ### setup model and optimizer
        # make new model
        from vae import get_VAE
        VAE = get_VAE(self.learn_force)
        self.model = VAE(**args['model_dict']).to(device=self.device,dtype=self.dtype)
        self.model.device = self.device
        self.model.dtype = self.dtype
        torch.set_default_dtype(self.dtype)
        self.model.eval()
        torch.set_num_threads(1)
        num_threads = get_num_cores()
        possible_chunks=np.arange(1,num_threads+1)
        chunks=int(possible_chunks[(self.num_pts % possible_chunks) == 0][-1] )
        self.model.build_chunk_decoder(chunks)
        # self.model.build_z_buffer()

        self.got_shared_model =  'shared_model' in args.keys()
        if self.got_shared_model:
            self.shared_model = args['shared_model']
        else: 
            self.shared_model = None

        ### setup clustering
        self.plot_lims = np.array(self.robot_lim)[self.plot_idx]
        self.last_clusters = None
        self.cluster_log = []
        self.cluster_names = ['step','error','num_clusters','clusters','stable?']
        self.clustering_config = {
            'model': str(self.model).split('\n'),
            'states': self.states,
            'plot_idx': self.plot_idx,
            'tray_lim': self.tray_lim,
            'robot_lim': self.robot_lim,
            'robot_ctrl_lim': self.robot_ctrl_lim,
            'num_pts': self.num_pts,
            'scale': 1.15,
            'uniform': True,
            'print_debug': False,
            'num_fingerprints': None,       # not used if cluster_method = mean_shift
            'get_blank': False,
            'cluster_by_plot_idx': False,    # not used if optimize_samples = True
            'cluster_weight': False,
            'batch': None,
            'sample_method': 'reweight',
            'cluster_method': 'mean_shift',
            'optimize_samples': True,
            'reweight_sample_scale': 2,
        }

        # set up figures
        self.fig = None
        self.save_path = self.dir_path + 'clusters/'
        if os.path.exists(self.save_path) == False:
            os.makedirs(self.save_path)

        with open(self.save_path+'cluster_config.yaml', 'w') as outfile:
            yaml.dump(self.clustering_config, outfile, default_flow_style=False)
        self.clustering_config['model'] = self.model 

    def map_dict(self, user_info):
        for k, v in user_info.items():
            setattr(self, k, v)

    @torch.no_grad()
    def load_model(self):
        PATH=self.dir_path+'clustering_model_checkpoint_tmp.pth'
        if not(self.got_shared_model) and os.path.exists(self.dir_path+'clustering_model_ready'):
            try:
                tmp = torch.load(PATH)
                self.model.load_state_dict(tmp['model'],strict=False)
                self.learning_ind = tmp['epoch']
                os.remove(self.dir_path+'clustering_model_ready')
                if self.print_debug: cprint('[CLUSTERING] model loaded','magenta')
            except:
                pass
        elif self.got_shared_model:
            self.model.load_state_dict(copy.deepcopy(self.shared_model.state_dict()),strict=False)
            self.learning_ind = self.shared_model.learning_ind.item()
            if self.print_debug: cprint('[CLUSTERING] model loaded','magenta')

    @torch.no_grad()
    def save_checkpoint(self):
        mod = f"_{self.model.learning_ind.item()}steps_cluster_checkpoint"
        torch.save(self.model.state_dict(), self.dir_path+'model_final'+mod+'.pth')  # state dict only

    def update(self,explr_step,save=True): 
        ## load model
        self.load_model()

        ## find clusters
        X,Y_labels,cluster_means,cluster_covariances,cov_type,samples,plot_mean_meas,msgs = find_clusters(**self.clustering_config)
                
        if self.print_debug: cprint('[CLUSTERING] plotting','magenta')

        ## plot
        plot_data = [explr_step,self.learning_ind,samples,plot_mean_meas,X,Y_labels,cluster_means,cluster_covariances,self.last_clusters]
        self.draw_fig(plot_data,save)

        ## add to running list
        if self.last_clusters is not None:
            num_clusters = len(cluster_means)
            if num_clusters == len(self.last_clusters):
                # get all permutations 
                all_combos = np.stack(list(it.product(np.stack(list(it.permutations(cluster_means))),np.stack(list(it.permutations(self.last_clusters))))))
                # find min error
                sq_diff = (all_combos[:,0] - all_combos[:,1])**2
                all_error = np.sum(sq_diff,axis=(1,2))/num_clusters
                error = np.min(all_error)
                # error = np.sum(( cluster_means - self.last_clusters)**2)/num_clusters
                stable = error < 0.001
                if stable: # check done condition
                    self.save_checkpoint()
            else: 
                error = 'NA'
                stable = False
            self.cluster_log.append([self.learning_ind,error,num_clusters,cluster_means,stable])            

        self.last_clusters = cluster_means
        if self.print_debug: cprint('[CLUSTERING] done with update','magenta')

    def draw_fig(self,plot_data,save=True,vert=True):
        explr_step,learning_ind,samples,dist,X,Y_labels,cluster_means,cluster_covariances,last_clusters = plot_data

        ## build fig
        if self.fig is None:
            if vert:
                self.fig,self.axs = plt.subplots(2,1,figsize=(3,6))
            else:
                self.fig,self.axs = plt.subplots(1,2,figsize=(6,3))
            if self.render_figs:
                plt.show(block=False)
        [ax.cla() for ax in self.axs]
        self.fig.suptitle(learning_ind)
        colors = sns.color_palette("Paired")

        ## draw plots
        num_fingerprints = len(cluster_means)
        plot_gmm_results(X, Y_labels, cluster_means, cluster_covariances, self.axs[0],color_iter=colors,title="New Cluster(s)",plot_idx=self.plot_idx)
        ax = self.axs[1]
        heatmap = ax.tricontourf(*samples[:,self.plot_idx].T,dist,cmap='gist_heat',levels=10)
        for c in heatmap.collections:
            c.set_edgecolor("face")
            c.set_rasterized(True)
        ax.set_title(f'Cluster Comparison')
        if last_clusters is not None:
            [ax.scatter(*mu[self.plot_idx],color=color,edgecolor='white',marker='^',s=200,label=f'old | {idx}',clip_on=False) for idx,(color,mu) in enumerate(zip(colors,last_clusters))]
        [ax.scatter(*mu[self.plot_idx],color=color,edgecolor='white',marker='o',s=200,label=f'new | {idx}',clip_on=False) for idx,(color,mu) in enumerate(zip(colors,cluster_means))]
        if vert:
            ax.legend(loc='center', bbox_to_anchor=(0.5, 1.25), ncol=2)
        else:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.9), ncol=2)
        for ax in self.axs:
            ax.set_aspect('equal', 'box')
            ax.set_xlim(self.plot_lims[0]*1.15)
            ax.set_ylim(self.plot_lims[1]*1.15)
            ax.xaxis.set_ticks(np.linspace(*self.plot_lims[0],5))
            ax.yaxis.set_ticks(np.linspace(*self.plot_lims[1],5))
            plot_label = [p if p == p.lower() else 'd' + p.lower() + '\dt' for p in self.plot_states]
            ax.set_xlabel(plot_label[0])
            ax.set_ylabel(plot_label[1])
        self.fig.tight_layout()

        if self.render_figs:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(1)

        ## save fig
        if self.save_figs and save:
            self.fig.savefig(self.save_path+f'clusters_explrStep{explr_step:05d}_learningIter{learning_ind:05d}.svg')

def cluster(rank,world_size,args,seed,data_queue):
    killer = setup(rank,world_size,seed,args['use_gpu'])
    clustering = Clustering(args)
    cprint(f'[CLUSTERING {rank}] initialized','magenta')

    done = False
    while not killer.kill_now and not done:
        try:
            if data_queue.poll(timeout=1):
                explr_step,done,save = data_queue.recv()
                if done:
                    cprint(f'[CLUSTERING {rank}] got done','yellow')
                    break
                clustering.update(explr_step,save)
            if clustering.fig is not None and clustering.fig.stale:
                clustering.fig.canvas.draw_idle()
        except (BrokenPipeError) as c:
            # cprint(f'[CLUSTERING {rank}] ERROR {c}','cyan')
            break
        except (RuntimeError) as c:
            cprint(f'[CLUSTERING {rank}] ERROR {c}','cyan')
            break

    cprint(f'[CLUSTERING {rank}] shutdown','yellow')

    # save log
    import pandas as pd
    pd.DataFrame(clustering.cluster_log,columns=clustering.cluster_names).to_csv(clustering.save_path + 'cluster_log.csv',index=False)
    clustering.fig.savefig(clustering.save_path+f'clusters_final.svg')
