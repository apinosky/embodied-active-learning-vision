
#!/usr/bin/env python

########## set these enviroment params ##########
save_checkpoint_models = False

########## global imports ##########
import torch
from torch import optim
from torch.distributions import Normal
from torch.nn.parallel import DistributedDataParallel as DDP

from termcolor import cprint
import time
import sys, os
import numpy as np
import datetime
import pickle
import math
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def smooth_data(locs,vals):
    if len(vals) < 5:
        return locs, vals
    try: 
        new_locs = np.linspace(locs[0], locs[-1], len(locs)*2)
        spl = make_interp_spline(locs, vals, k=3)
        smooth = spl(new_locs)
        return new_locs, smooth
    except: 
        print(len(locs),len(vals))
        return locs,vals

########## local imports ##########
from plotting.plotting_matplotlib import set_mpl_format
from control_torch.klerg_utils import traj_footprint_vec, renormalize, traj_spread_vec

set_mpl_format()

# --------- Trainer --------------------

def get_loss(y,y_pred,y_logvar,z_mu,z_logvar):
    y_logvar = y_logvar.expand_as(y_pred)
    ### using torch.distributions.Normal / log_prob
    # q = Normal(y_pred, y_logvar.exp())
    # RC_loss = - torch.mean(q.log_prob(y))
    ### using hard coded log_prob
    var = (y_logvar.exp() ** 2)
    log_prob = -((y - y_pred) ** 2) / (2 * var) - y_logvar - math.log(math.sqrt(2 * math.pi))
    RC_loss = - torch.mean(log_prob)
    KL_loss = - torch.mean(0.5 * (1 + z_logvar - z_mu**2 - z_logvar.exp()).sum(1))
    return RC_loss,KL_loss

@torch.no_grad()
def update_loss_plots(plot_losses,plot_combo,trainer,fig_data,plot_spread=False): 
    init = fig_data is None
    if not init:
        loss_fig_data, spread_fig_data = fig_data
    if plot_losses:
        if init:
            loss_fig,loss_ax = plt.subplots(2,1)
            losses = [trainer.KL_losses,trainer.RC_losses,trainer.losses,trainer.z_activity,trainer.active_units]
            losses_labels = ['KL_losses','RC_losses','total loss']

            lines = [loss_ax[0].plot(0,0,label = l)[0] for l in losses_labels]
            loss_ax[0].legend()
            loss_ax[0].set_ylabel('trainer losses')
            lines = lines + [loss_ax[1].plot(0,0,label = l)[0] for l in ['activity','units']]
            loss_ax[1].set_xlabel('update steps')
            loss_ax[1].set_ylabel('latent space')
            loss_ax[1].legend()
            if trainer.render_figs:
                plt.ion()
                plt.show(block=False)
            loss_fig_data = [loss_fig,loss_ax,lines,losses]
        else: 
            loss_fig,loss_ax,lines,losses = loss_fig_data
        for l,data in zip(lines,losses):
            l.set_data(np.arange(len(data)),data)
        loss_fig.canvas.blit(loss_fig.bbox)
        loss_fig.canvas.flush_events()
        for ax in loss_ax:
            ax.relim()
            ax.autoscale_view()
    elif plot_combo:
        if init:
            loss_fig,loss_axes = plt.subplots(4,1,figsize=(5,4*2))

            # move figure
            try:
                loss_fig.canvas.manager.window.move(0,0)
            except:
                pass

            activity_ax = loss_axes[0]
            activity = [trainer.active_units]
            lines = [activity_ax.plot(0,0,label = l)[0] for l in ['active units']]
            # activity = [trainer.z_activity,trainer.active_units,trainer.active_units_vars]
            # lines = [activity_ax.plot(0,0,label = l)[0] for l in ['activity','units (mean)','units (var)']]

            activity_ax.set_xlabel('update steps')
            activity_ax.set_ylabel('latent space')
            activity_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=3, columnspacing=0.2)


            new_explr_step,new_grade,new_spread = trainer.replay_buffer.get_hyperparams()
            new_xi = trainer.replay_buffer.get_xi()
            explr_step = [new_explr_step]
            spread = [new_spread]
            grade = [new_grade]
            xi = [new_xi]

            loss_ax = loss_axes[1]
            llabels = ['KL_losses','RC_losses']
            llosses = [trainer.KL_losses,trainer.RC_losses]
            if trainer.other_locs:
                llabels = llabels  + ['RC_o_losses']
                llosses = llosses  + [trainer.RC_o_losses]
            llabels = llabels + ['total loss']
            llosses = llosses + [trainer.losses]
            l2 = loss_ax.twinx()
            l2_color = 'tab:blue'
            l2.yaxis.label.set_color(l2_color) 
            l2.spines['right'].set_color(l2_color)
            l2.tick_params(axis='y', colors=l2_color)
            loss_axes = np.hstack([loss_axes,l2])
            lines.append(l2.plot(0,0,label = llabels[0],color=l2_color)[0])
            lines = lines + [loss_ax.plot(0,0,label = l)[0] for l in llabels][1:]
            loss_ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=len(llosses), columnspacing=0.2)
            loss_ax.set_xlabel('update steps')
            loss_ax.set_ylabel('trainer losses')

            spread_ax = loss_axes[2]
            lines.append(spread_ax.plot(explr_step,spread)[0])
            spread_ax.set_xlabel('explr steps')
            spread_ax.set_ylabel('spread')

            grade_ax = loss_axes[3]
            lines.append(grade_ax.plot(explr_step,grade)[0])
            grade_ax.set_xlabel('explr steps')
            grade_ax.set_ylabel('grade')
            # g2 = grade_ax.twinx()
            # g2_color = 'tab:green'
            # g2.yaxis.label.set_color(g2_color) 
            # g2.spines['right'].set_color(g2_color)
            # g2.tick_params(axis='y', colors=g2_color)
            # loss_axes = np.hstack([loss_axes,g2])
            # lines.append(g2.plot(explr_step,xi,color=g2_color)[0])
            # g2.set_ylabel('xi')

            losses = activity + llosses + [spread,grade,xi] 
            steps = [False]*len(activity) +  [False]*len(llosses) + [True,True] #,True] 

            loss_fig.tight_layout()
            if trainer.render_figs:
                plt.ion()
                plt.show(block=False)
            loss_fig_data = [loss_fig,loss_axes,lines,losses,steps,explr_step]
        else: 
            loss_fig,loss_axes,lines,losses,steps,explr_step = loss_fig_data
            new_explr_step,new_grade,new_spread = trainer.replay_buffer.get_hyperparams()
            new_xi = trainer.replay_buffer.get_xi()
            explr_step.append(new_explr_step)
            losses[-3].append(new_spread)
            losses[-2].append(new_grade)
            losses[-1].append(new_xi)


        for l,data,use_explr_steps in zip(lines,losses,steps):
            locs = explr_step if use_explr_steps else np.arange(len(data))
            # l.set_data(*smooth_data(locs,data))
            l.set_data(locs,data)
        loss_fig.canvas.blit(loss_fig.bbox)
        loss_fig.canvas.flush_events()
        for ax in loss_axes:
            ax.relim()
            ax.autoscale_view()
    else: 
        loss_fig_data = []
    if plot_spread: 
        if init:
            spread_fig,spread_ax = plt.subplots(1,1,figsize=(3,3))
            spread_ax.set_title('spread')
        else:
            spread_fig,spread_ax,tri = spread_fig_data
            if hasattr(tri,'collections'):
                # removes only the contours, leaves the rest intact
                [c.remove() for c in tri.collections]

        tri = spread_ax.tricontourf(*trainer.plotting_log['samples'][:,trainer.plot_idx].T,trainer.plotting_log['max_q'],levels=30,cmap='gist_heat')
        spread_fig_data = [spread_fig,spread_ax,tri]
    else:
        spread_fig_data = []
    return [loss_fig_data,spread_fig_data]

class Trainer(object):
    def __init__(self,args,rank,killer):
        ## load pararms
        self.map_dict(vars(args['args']))
        self.map_dict(args['model_dict'])
        self.rank = rank
        self.killer = killer
        self.iter = 0

        ### setup model and optimizer
        # make new model
        from vae import get_VAE
        VAE = get_VAE(self.learn_force)
        model = VAE(**args['model_dict']).to(device=self.device,dtype=self.dtype)
        model.device = self.device
        model.dtype = self.dtype
        torch.set_default_dtype(self.dtype)
        # model.share_memory() # try to force into RAM
        optimizer = optim.Adam(model.parameters(),lr=self.model_lr) # /self.num_learning_opt
        # model = model.to(memory_format=torch.channels_last) # ipex does this automatrically
        model.train()
        var = model.reshape_var(torch.rand(self.batch_per_proc,model.ylogvar_dim)) # for jit warmup
        y_dim = self.img_dim
        # model = torch.jit.script(model)
        self.model = model
        if self.ddp_trainer:
            self.model = DDP(self.model,static_graph=True) #,bucket_cap_mb=5000)
        self.optimizer = optimizer
        self.replay_buffer = args['replay_buffer']

        ### setup learning params]
        for check_key, default_val in zip(['target_samples_scale','gamma_weight'],[1.,0.1]):
            if not hasattr(self,check_key):
                setattr(self,check_key,default_val)
        for check_key, default_val in zip(['fixed_beta','fixed_gamma'],[False,False]):
            if not(check_key in self.hyperparam_ramp.keys()):
                self.hyperparam_ramp[check_key] = default_val

        # self.num_target_samples = int(self.num_target_samples*self.target_samples_scale)
        self.std = self.std / self.target_samples_scale
        self.fixed_beta = self.hyperparam_ramp['fixed_beta']
        self.entropy_based_beta = (not self.fixed_beta) and (not self.hyperparam_ramp['beta_manual_ramp']) 
        self.fixed_gamma = self.hyperparam_ramp['fixed_gamma']
        self.entropy_based_gamma = (not self.fixed_gamma) and (not self.hyperparam_ramp['gamma_manual_ramp']) 
        self.grade = 0.
        self.spread = 0.
        if self.fixed_beta:
            self.beta = self.beta_start_weight 
        elif self.entropy_based_beta: 
            self.beta = 0.
        else: 
            self.map_dict(self.hyperparam_ramp)
            self.beta = self.beta_start_weight # controls the importance of keeping the information encoded in z close to the prior. 0 corresponds to a vanilla autoencoder, and 1 to the correct VAE objective.
            self.d_beta = (self.beta_end_weight - self.beta_start_weight)/max(self.beta_warmup_steps,1)
            self.beta_warmup_epoch = max(self.beta_warmup_epoch,1)
        if self.fixed_gamma:
            self.gamma = self.gamma_start_weight 
        elif self.entropy_based_gamma or not(self.other_locs): 
            self.gamma = 0.
        else: 
            if self.entropy_based_beta: self.map_dict(self.hyperparam_ramp)
            self.gamma = self.gamma_start_weight # 0 corresponds to a VAE objective.
            self.d_gamma = (self.gamma_end_weight - self.gamma_start_weight)/max(self.gamma_warmup_steps,1)
            self.gamma_warmup_epoch = max(self.gamma_warmup_epoch,1)
        # self.use_next_state = False

        ## setup plotting
        if self.rank == 0:
            if os.path.exists(self.dir_path) == False:
                os.makedirs(self.dir_path)
            self.log_file = "log.txt"
            self.start_time = time.time()
            self.losses = []
            self.KL_losses = []
            self.RC_losses = []
            self.z_activity = []
            self.active_units = []
            self.z_activity_vars = []
            self.active_units_vars = []
            self.beta_log = []
            self.grade_log = []
            self.spread_log = []
            self.xi_log = []
            self.plotting_log = {'samples':None,'max_q':None,'entropy_dist':None}

            # self.loss_mean = torch.nn.MSELoss(reduction='mean')
            self.learning_ind = 0
            self.training_update = None
            self.checkpoint_update = None
            self.init_checkpoint = False

            if self.other_locs: 
                self.RC_o_losses = []
                self.gamma_log = []

            # every iter
            self.x_eval_last = torch.zeros(1,self.s_dim,device=self.device,dtype=self.dtype)
            self.y_eval_last = torch.zeros(1,*y_dim,device=self.device,dtype=self.dtype)
            self.force_eval_last = torch.zeros(1,1,device=self.device,dtype=self.dtype)
            # every 10 iter
            self.x_eval_checkpoint = torch.zeros(1,self.s_dim,device=self.device,dtype=self.dtype)
            self.y_eval_checkpoint = torch.zeros(1,*y_dim,device=self.device,dtype=self.dtype)
            self.force_eval_checkpoint = torch.zeros(1,1,device=self.device,dtype=self.dtype)

            if 'shared_model' in args.keys():
                self.shared_model = args['shared_model']
        if self.print_debug: cprint(f'[TRAINER {self.rank}] {torch.get_rng_state().sum()} {torch.randn(1)}','blue') # check seed is initially the same

        # warmup jit
        dummy_inputs = [torch.rand(self.batch_per_proc,*y_dim),
                        torch.rand(self.batch_per_proc,*y_dim),
                        var,
                        torch.rand(self.batch_per_proc,self.z_dim),
                        torch.rand(self.batch_per_proc,self.z_dim)]
        self.get_loss_jit = torch.jit.script(get_loss) #,dummy_inputs)
        for _ in range(5):
            self.get_loss_jit(*dummy_inputs)

        # set up uniform sampler
        if self.rank == 0:
            explr_idx = list(range(len(self.states)))
            lims = torch.as_tensor(self.robot_lim[explr_idx],dtype=self.dtype,device=self.device)
            self.state_sampler = torch.distributions.uniform.Uniform(*lims.T)
            self.img_dim = y_dim

    def build_chunk_decoder(self,num_threads=20):
        if self.rank == 0: 
            if self.ddp_trainer: 
                self.model.module.build_chunk_decoder(num_threads) 
            else:
                self.model.build_chunk_decoder(num_threads) 

    @torch.no_grad()
    def process_buffer(self,single=False,weighted=False,other_locs=False,last=False): 
        if last:
            buff = self.replay_buffer.get_last()
        elif single:
            buff = self.replay_buffer.sample(1,weighted=weighted)
        else:
            buff = self.replay_buffer.sample_batch(rank=self.rank,weighted=weighted)
        x,y = buff[0].to(device=self.device),buff[1].to(device=self.device)
        if self.learn_force:
            force = buff[2].to(device=self.device)
        else:
            force = None
        force2 = None

        if other_locs:
            if single: 
                buff2 = self.replay_buffer.sample(1,weighted=False)
            else: 
                buff2 = self.replay_buffer.sample_batch(rank=self.rank,weighted=False)
            x2,y2 = buff2[0].to(device=self.device),buff2[1].to(device=self.device)
            if self.learn_force: 
                force2 = buff2[2].to(device=self.device)
            if self.dx: 
                x2 = x2 - x
        else: 
            x2 = torch.empty(0)
            y2 = None
        
        return x,y,force,x2,y2,force2

    def process_model(self,x,y,force,x_decode=torch.empty(0)): 
        if self.learn_force:
            out = self.model(x,y,force,x_decode)
            y_pred, y_logvar, z_mu, z_logvar, z_samples, force_pred, force_logvar, img_pred_decode, img_logvar_decode, force_pred_decode, force_pred_logvar = out            
        else:
            out = self.model(x,y,x_decode)
            y_pred, y_logvar, z_mu, z_logvar, z_samples, img_pred_decode, img_logvar_decode = out
            force_pred = None
            force_logvar = None
            force_pred_decode = None
            force_pred_logvar = None
        return y_pred, y_logvar, z_mu, z_logvar, force_pred, force_logvar, img_pred_decode, img_logvar_decode, force_pred_decode, force_pred_logvar

    def __call__(self,weighted=False):
        self.model.train()
        with torch.no_grad():
            # try messing with beta another way 
            _,self.grade,self.spread = self.replay_buffer.get_hyperparams()
            if self.entropy_based_beta: 
                self.beta = self.grade
            if self.other_locs and self.entropy_based_gamma: 
                self.gamma = self.spread
        # Run VAE Optimization
        step_loss = []
        for idx in range(self.num_learning_opt):
            ## check for batch 
            ready = False
            while not ready:
                ready = self.replay_buffer.check_batch(self.rank)
                if self.killer is not None and self.killer.kill_now:
                    break
            if self.killer is not None and self.killer.kill_now:
                break
            ## get data
            x,y,force,x2,y2,force2 = self.process_buffer(weighted=weighted,other_locs=self.other_locs)

            if self.print_debug: cprint(f'[TRAINER {self.rank}] got batch data,iter {self.iter} position {self.replay_buffer.position}','yellow')
            if self.print_debug: cprint(f'[TRAINER {self.rank}] {torch.get_rng_state().sum()} {torch.randn(1)}','blue') # check seed changes
            # Run optimization
            y_pred, y_logvar, z_mu, z_logvar, force_pred, force_logvar, y_pred2, y_logvar2, force_pred2, force_logvar2 = self.process_model(x,y,force,x2)

            ## get losses
            RC_loss,KL_loss = self.get_loss_jit(y,y_pred, y_logvar, z_mu, z_logvar)
            vae_loss = RC_loss + self.beta*KL_loss

            if self.learn_force:
                force_loss,_ = self.get_loss_jit(force,force_pred, force_logvar, z_mu, z_logvar)
            else:
                force_loss = 0.

            if self.other_locs: # and self.iter >= 1000:
                RC_loss_o,_ = self.get_loss_jit(y2,y_pred2, y_logvar2, z_mu, z_logvar)
                other_loss = self.gamma*(RC_loss_o)
                if self.learn_force:
                    force_loss_o,_ = self.get_loss_jit(force2,force_pred2, force_logvar2, z_mu, z_logvar)
                    other_loss += self.gamma*(force_loss_o)
            else:
                RC_loss_o = torch.zeros(1) # placeholder
                other_loss = 0.

            loss = vae_loss + force_loss + other_loss*self.gamma_weight

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                step_loss.append(loss.detach())

                if self.rank == 0:
                    self.beta_log.append(self.beta)
                    self.grade_log.append(self.grade)
                    self.spread_log.append(self.spread)
                    self.xi_log.append(self.xi)
                    if self.other_locs:
                        self.gamma_log.append(self.gamma)
                        self.RC_o_losses.append(RC_loss_o.item())
                    self.KL_losses.append(KL_loss.item())
                    self.RC_losses.append(RC_loss.item())
                    # entropy = torch.mean( 0.5 * (z_logvar_au.exp().sum(1).log()))
                    # self.z_entropy.append(entropy.item())
                    threshold = 0.01
                    vars_of_means = torch.var(z_mu,axis=0)
                    means_of_vars = torch.mean(z_logvar.exp(),axis=0)
                    active_units = (vars_of_means > threshold)
                    active_units_vars = (means_of_vars < threshold)
                    self.z_activity.append(vars_of_means.sum().item())
                    # self.z_activity.append(vars_of_means[active_units_vars].sum().item())
                    self.active_units.append(active_units.sum().item())
                    self.active_units_vars.append(active_units_vars.sum().item())

                if not self.entropy_based_beta and not self.fixed_beta: 
                    if (self.iter < self.beta_warmup_steps * self.beta_warmup_epoch) and (self.iter % self.beta_warmup_epoch) == 0:
                        self.beta += self.d_beta
                    elif self.rank==0 and  (self.iter == self.beta_warmup_steps * self.beta_warmup_epoch):
                        cprint(f"TRAINER {self.rank} | finished ramping up beta",'green')
                if self.other_locs and not self.entropy_based_gamma and not self.fixed_gamma: 
                    if (self.iter < self.gamma_warmup_steps * self.gamma_warmup_epoch) and (self.iter % self.gamma_warmup_epoch) == 0:
                        self.gamma += self.d_gamma
                    elif self.rank==0 and  (self.iter == self.gamma_warmup_steps * self.gamma_warmup_epoch):
                        cprint(f"TRAINER {self.rank} | finished ramping up gamma",'green')
                self.iter += 1

        if self.print_debug: cprint(f"TRAINER {self.rank} | step {self.optimizer.state_dict()['state'][0]['step']}",'blue')
        return torch.stack(step_loss)

    def map_dict(self, user_info):
        for k, v in user_info.items():
            setattr(self, k, v)

    def write_to_log(self,msg):
        print(msg)
        with open(self.dir_path + self.log_file,"a") as f:
            f.write(msg+'\n')

    @property
    def duration_str(self):
        return str(datetime.timedelta(seconds=(time.time()-self.start_time)))

    @torch.no_grad()
    def get_learning_ratio(self,learning_step,explr_step): 
        denom = (explr_step - self.frames_before_training)
        if denom == 0: 
            return learning_step
        else: 
            return learning_step / denom

    @torch.no_grad()
    def pre_train_mp(self,explr_step,last=True):
        x_r,y_r,force_r,_,_,_ = self.process_buffer(single=True,last=last,weighted=False,other_locs=False)
        self.x_eval_last.copy_(x_r,non_blocking=True)
        self.y_eval_last.copy_(y_r,non_blocking=True)
        if self.learn_force:
            self.force_eval_last.copy_(force_r,non_blocking=True)
        self.pre_iter_last = self.process_model(self.x_eval_last, self.y_eval_last,self.force_eval_last)[0] # y_pred
        if not self.init_checkpoint:
            msg = 'explr_step learning_step ratio loss \t| runtime'
            self.write_to_log(msg)
            fname = self.dir_path + f'iter{self.learning_ind}'
            self.init_checkpoint = True
            self.x_eval_checkpoint.copy_(x_r,non_blocking=True)
            self.y_eval_checkpoint.copy_(y_r,non_blocking=True)
            if self.learn_force:
                self.force_eval_checkpoint.copy_(force_r,non_blocking=True)
            self.pre_iter_checkpoint = self.process_model(self.x_eval_checkpoint, self.y_eval_checkpoint, self.force_eval_checkpoint)[0] # y_pred
            # if self.entropy_based_beta:
            #     if self.ddp_trainer: 
            #         self.model.module.build_z_buffer() 
            #     else:
            #         self.model.build_z_buffer() 

        samples = self.state_sampler.sample((self.num_target_samples,))

        # get "spread" from buffer
        dim = samples.shape[1]
        explr_idx = torch.arange(dim,device=self.device)
        traj = self.replay_buffer.get_all_x()
        std = torch.tensor([self.std]*dim,device=self.device)
        max_q = traj_spread_vec(traj, samples, explr_idx, std, nu=1.)
        max_q /= torch.max(max_q)
        spread = max_q.mean()

        # get "grade" from conditional entropy
        inputs = [x_r,y_r]
        if self.learn_force:
            inputs.append(force_r)
        if self.ddp_trainer:
            self.model.module.update_dist(*inputs)
            entropy_dist = self.model.module.pdf_torch(samples)
        else:
            self.model.update_dist(*inputs)
            entropy_dist = self.model.pdf_torch(samples)
        entropy_dist = entropy_dist**spread 
        entropy_dist /= entropy_dist.max()
        # entropy_dist = renormalize(entropy_dist)

        # self.xi = self.replay_buffer.get_xi() 
        self.xi = 4 # 3.5

        grade = torch.clamp(10.**(-torch.log10(entropy_dist.min())-self.xi),max=0.01) # min entropy

        self.plotting_log['samples'] = samples.clone().cpu()
        self.plotting_log['max_q'] = max_q.clone().cpu()
        self.plotting_log['entropy_dist'] = entropy_dist.clone().cpu()

        self.replay_buffer.update_hyperparams(explr_step,grade,spread)

    @torch.no_grad()
    def post_train_mp(self,explr_step,losses,print_update=50,start_ind=None,plot=True,last=True,unweighted_count=None):
        if self.use_gpu:
            torch.cuda.synchronize()
        self.model.eval()

        # update logs
        if losses is not None:
            start_ind = self.learning_ind
            self.losses += losses.tolist()
            self.learning_ind += len(losses)
        learning_steps = np.arange(start_ind,self.learning_ind)

        # print(start_ind,self.duration_str)
        if start_ind > 0:
            if any(learning_steps % print_update == 0):
                update_step = learning_steps[learning_steps % print_update == 0]
                for learning_ind in update_step:
                    ratio = self.get_learning_ratio(learning_ind,explr_step)
                    if unweighted_count is not None: 
                        weighted_ratio = self.get_learning_ratio(learning_ind-unweighted_count,explr_step)
                        ratio = '{:0.1f}/{:0.1f}'.format(ratio,weighted_ratio)
                    else: 
                        ratio = '{:0.1f}'.format(ratio)
                    if losses is not None:
                        msg = '{}\t {}\t {}\t {:0.3f}\t | {}'.format(explr_step, learning_ind, ratio, sum(self.losses[learning_ind-50:learning_ind]), self.duration_str)
                    else:
                        msg = '{}\t {}\t  {}\t | {}'.format(explr_step, learning_ind, ratio, self.duration_str)
                    self.write_to_log(msg)
            if any(learning_steps % self.save_rate == 0):
                fpath = self.dir_path+ 'model_checkpoint_iter{}.pth'.format(self.learning_ind)
                if save_checkpoint_models:
                    cp_dict = {
                            'iter': self.learning_ind,
                            'state_dict': self.model.module.state_dict() if self.ddp_trainer else self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict() if self.optimizer is not None else 'None',
                            'duration': self.duration_str,
                            'losses' : self.losses
                    }
                    torch.save(cp_dict, fpath)

                if plot: 
                    self.fname =  'iter{:05d}'.format(self.learning_ind)
                    # plotting
                    post_checkpoint = self.process_model(self.x_eval_checkpoint, self.y_eval_checkpoint,self.force_eval_checkpoint)[0] # y_pred
                    self.checkpoint_update = [self.y_eval_checkpoint[0].detach().clone().permute(2,1,0),
                                            self.pre_iter_checkpoint[0].detach().clone().permute(2,1,0),
                                            post_checkpoint[0].detach().clone().permute(2,1,0),
                                            [self.learning_ind,self.save_rate]]

                    # Sample Buffer
                    x,y,force,_,_,_ = self.process_buffer(single=True,last=last,weighted=False,other_locs=False)
                    self.x_eval_checkpoint.copy_(x,non_blocking=True)
                    self.y_eval_checkpoint.copy_(y,non_blocking=True)
                    if self.learn_force:
                        self.force_eval_checkpoint.copy_(force,non_blocking=True)
                    self.pre_iter_checkpoint = self.process_model(self.x_eval_checkpoint, self.y_eval_checkpoint,self.force_eval_checkpoint)[0] # y_pred

            # plotting
            post_iter_img_pred = self.process_model(self.x_eval_last,self.y_eval_last,self.force_eval_last)[0] # y_pred

            self.training_update = [self.y_eval_last[0].detach().clone().permute(2,1,0),
                                    self.pre_iter_last[0].detach().clone().permute(2,1,0),
                                    post_iter_img_pred[0].detach().clone().permute(2,1,0),
                                    [self.learning_ind,self.num_learning_opt]]

    @torch.no_grad()
    def save_checkpoint(self):
        if self.rank == 0:
            if self.print_debug: cprint(f'[TRAINER {self.rank}] saving model','yellow')
            if self.use_gpu:
                torch.cuda.synchronize()
            if hasattr(self,'shared_model'):
                state_dict = copy.deepcopy(self.model.module.state_dict()) if self.ddp_trainer else copy.deepcopy(self.model.state_dict())
                if self.use_gpu:
                    state_dict = {k: v.cpu() for k, v in state_dict.items()}
                self.shared_model.load_state_dict(state_dict,strict=False)
                self.shared_model.learning_ind[0] = self.iter
                # self.shared_model.to('cpu')
            else:
                tmp = {'model':self.model.module.state_dict() if self.ddp_trainer else self.model.state_dict(),
                'epoch':self.learning_ind}
                torch.save(tmp, self.dir_path+'model_checkpoint_tmp.pth')
                with open(self.dir_path+'model_ready','w') as f:
                    pass

    @torch.no_grad()
    def save(self,post_explr=False,callback=False,mod=""):
        # Save Pickled Data
        data_eval_dict = {
            'iter': self.learning_ind,
            'duration': self.duration_str,
            'losses' : self.losses,
            'KL_loss' : self.KL_losses,
            'RC_loss' : self.RC_losses,
            'active_units' : self.active_units,
            'z_activity' : self.z_activity,
            'active_units_vars' : self.active_units_vars,
            'beta_log' : self.beta_log,
            'spread_log' : self.spread_log,
            'grade_log' : self.grade_log,
            }
        if self.other_locs:
            data_eval_dict['gamma_log'] = self.gamma_log
            data_eval_dict['RC_o_loss'] = self.RC_o_losses
            
        pickle.dump( data_eval_dict, open( self.dir_path+"data_eval_dict_trainer.pickle", "wb" ), protocol=pickle.HIGHEST_PROTOCOL )

        # Save Torch final model
        if mod == "":
            if post_explr:
                mod = "_postexplr"
                self.write_to_log(f'final runtime: {self.duration_str}')
            elif callback:
                mod = "_callback"
            else:
                self.write_to_log(f'explr_learning runtime: {self.duration_str}')
        state_dict = copy.deepcopy(self.model.module.state_dict()) if self.ddp_trainer else copy.deepcopy(self.model.state_dict())
        if self.use_gpu:
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(state_dict, self.dir_path+'model'+mod+'.pth') # state dict only
        # torch.save(self.model.module, self.dir_path+'model'+mod+'.pth') # full model requires path continuity
        # torch.save(self.model, self.dir_path+'model'+mod+'.pth') # full model requires path continuity
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), self.dir_path+'optim'+mod+'.pth') # state dict only
            # torch.save(self.optimizer, self.dir_path+'optim'+mod+'.pth') # full model requires path continuity
