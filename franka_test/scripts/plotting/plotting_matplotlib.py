#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from franka.franka_utils import ws_conversion
import matplotlib.tri as mtri
# import scicomap as sc
from scipy.ndimage import gaussian_filter

## change default figure params
def set_mpl_format():
    try:
        font = 'Times New Roman'
        mpl.font_manager.findfont(font,fallback_to_default=False)
        plt.rcParams.update({'font.family':font})
        mpl.rcParams['toolbar'] = 'None'
    except:
        pass
set_mpl_format()
screen_extent = [0,0,1000,2000] # x y width height
try: 
    from screeninfo import get_monitors
    s = get_monitors()[0]
    screen_extent = [s.x,s.y,s.x+s.width,s.y+s.height] # x y width height
except: 
    pass

def move_figure(f, loc, x=0, y=0):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = mpl.get_backend()
    if 'left' in loc.lower():
        x += screen_extent[0]
    elif 'right' in loc.lower(): 
        x += screen_extent[2]
    if 'top' in loc.lower():
        y += screen_extent[1]
    elif 'bottom' in loc.lower():
        y += screen_extent[3]
        print(x,y)
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)

use_smoothed_dists = True

class Plotter(object):
    def __init__(self,plot_idx,xinit,render=True,path=None,plot_zs=True,states=None,robot_lim=None,tray_lim=None,save_folder='explr/',tdist=False):
        self.plot_idx = plot_idx 
        lim = robot_lim[plot_idx]

        # extra to deal with angles
        self.robot_lim = robot_lim
        self.tray_lim = tray_lim
        self.convert_angle = not(np.any(robot_lim == None)) and not(np.any(tray_lim == None))
        self.states = states
        self.interp = True
        self.use_imshow = False
        self.scatter = False
        if self.interp:
            num_samples = 50
            self.update_samples(lim,num_samples)
        self.plot_future_states = True

        # general
        self.render=render
        self.path = path + save_folder
        if os.path.exists(path+save_folder) == False:
            os.makedirs(path+save_folder)

        # set up plotting (fig 1)
        self.plot_zs = plot_zs
        if plot_zs:
            fig,ax = plt.subplots(2,3,figsize=(9,5.5))
        else:
            fig,ax = plt.subplots(2,2,figsize=(6,5.5))
        # ax[0,0].set_title('Sensor View')
        ax[0,0].set_ylabel('Sensor View',size='large')
        path_ax = ax[0,1]
        path_ax.set_title('Sensor Path')
        # path_ax.set_xlim(lim[0],lim[1])
        # path_ax.set_ylim(lim[0],lim[1])
        self.poses = []
        self.pose_history = path_ax.plot(xinit[0],xinit[1],"k.",clip_on=False,zorder=0)
        path_ax.plot(xinit[0],xinit[1],"r*",clip_on=False,zorder=1)
        self.current_pose = path_ax.plot(xinit[0],xinit[1],"gs",ms=12,clip_on=False,zorder=2)
        if self.plot_future_states:
            self.pose_plan = path_ax.plot(xinit[0],xinit[1],"b.",clip_on=False,zorder=3)
        self.min_p = ax[1,0].annotate(f"min: {1.:0.4f}", 1.75*lim[:,0],annotation_clip=False)
        ax[1,0].set_title('Target Distribution' if tdist else 'CVAE Conditional Entropy')
        ax[1,1].set_title('Time Averaged Distribution') # (q)')
        if plot_zs:
            ax[0,2].set_title(r'Latent Space $(\mu)$')
            ax[1,2].set_title(r'Latent Space $(\sigma^2)$')
        for idx,axs in enumerate(ax.ravel()):
            if idx == 0:
                # axs.axes.xaxis.set_visible(False)
                # axs.axes.yaxis.set_visible(False)
                axs.xaxis.set_ticklabels([])
                axs.xaxis.set_ticks([])
                axs.yaxis.set_ticklabels([])
                axs.yaxis.set_ticks([])
                axs.set_aspect('equal', adjustable='box')
            elif plot_zs and (idx in [2,5]):
                axs.set_ylabel('.\n.\n.')
                axs.yaxis.label.set_color('white')
            else:
                axs.set_xlim(lim[0,0],lim[0,1])
                axs.set_ylim(lim[1,0],lim[1,1])
                axs.axes.xaxis.set_ticks(np.linspace(lim[0,0],lim[0,1],5))
                axs.axes.yaxis.set_ticks(np.linspace(lim[1,0],lim[1,1],5))
                plot_label = [self.states[p] if self.states[p] == self.states[p].lower() else 'd' + self.states[p].lower() + '\dt' for p in self.plot_idx]
                axs.set_xlabel(plot_label[0])
                axs.set_ylabel(plot_label[1])
                axs.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        if self.render:
            move_figure(fig,'topRight')
            plt.ion()
            plt.show(block=False)
        fig.canvas.blit(fig.bbox)
        self.fig1 = fig
        self.ax1 = ax
        self.update_plot_dims = (4.,2.)

        self.cam_view = None
        self.z_figs = None
        self.cbar = None
        self.cmap = 'gist_heat' # sc.ScicoSequential('heat_r').get_mpl_color_map()
        mpl.style.use('fast')

        if self.use_imshow and self.interp:
            self.robot_figs = [self.ax1[1,0].imshow(np.zeros((num_samples,num_samples)),cmap=self.cmap,extent=(*lim[0],*lim[1]),origin='lower'),
                               self.ax1[1,1].imshow(np.zeros((num_samples,num_samples)),cmap=self.cmap,extent=(*lim[0],*lim[1]),origin='lower')]
        else:
            self.robot_figs = None

    def update_samples(self,lim,num_samples=50):
        self.xy_grid = np.meshgrid(*np.linspace(*lim.T, num_samples).T)
        # self.xy_grid = np.meshgrid(*[np.linspace(*lim, num_samples)]*2)
        self.robot_figs = None

    def update(self,args,proj_state=True,draw=True,throttle=False):
        [cam_data,state,force,robot_plot_data,z_mu,z_var,img_pred,iter_step]= args
          
        if self.cam_view is None:
            self.cam_view = self.ax1[0,0].imshow(cam_data,cmap='gray')
        else:
            self.cam_view.set_data(cam_data)
            self.cam_view.autoscale()
        if iter_step is not None:
            self.ax1[0,0].set_title(f'explore iteration: {iter_step[0]}')
            self.ax1[0,0].set_xlabel(f'learn iteration: {iter_step[1]}',size='large')
        if (self.states is not None) and ('z' in self.states):
            marker_size = 25.+(1+state[self.states.rfind('z')]).copy()*10. # resize based on z
        else:
            marker_size = 25.
        if proj_state:
            state = self.project_samples(state[None,:]).squeeze()
        self.poses.append(state[:2])
        self.pose_history[0].set_data(*np.array(self.poses).T)
        self.current_pose[0].set_data(state[0],state[1])
        if self.states is not None:
            self.current_pose[0].set_markersize(marker_size)
            self.current_pose[0].set_markeredgecolor('k')
            self.current_pose[0].set_markeredgewidth(min(force,5.))
        if robot_plot_data is not None:
            if use_smoothed_dists: 
                robot_dists = robot_plot_data[4:6]
            else:
                robot_dists = robot_plot_data[1:3]
            samples = self.project_samples(robot_plot_data[0])
            if len(self.states)>2: 
                targ_str = f"full min: {np.amin(robot_plot_data[1]):0.4f}, smoothed min:{np.amin(robot_plot_data[4]):0.4f}"
            else: 
                targ_str = f"min: {np.amin(robot_dists[0]):0.4f}"
            self.min_p.set_text(targ_str)
            if not throttle: 
                if not self.use_imshow:
                    if self.robot_figs is not None:
                        for cont in self.robot_figs:
                            for c in cont.collections:
                                c.remove()  # removes only the contours, leaves the rest intact
                    self.robot_figs = []

                if self.scatter:
                    _min, _max = np.amin(robot_dists), np.amax(robot_dists)
                    if self.robot_figs is None:
                        self.robot_figs=[ax.scatter( *samples.T, c=data, alpha=data, vmin=_min, vmax=_max, cmap=self.cmap) for ax,data in zip([self.ax1[1,0],self.ax1[1,1]],robot_dists)]
                    else: 
                        for ax,data in zip(self.robot_figs,robot_dists):
                            ax.set(clim=(_min,_max),alpha=data,array=data,offsets=samples[:,:2]) # array=color, offset=xy
                            ax.autoscale()

                elif self.interp:
                    # try interpolating samples befor passing to plotting function
                    with np.errstate(invalid='ignore'):
                        triang = mtri.Triangulation(*samples.T)
                        # zi = []
                        for idx,data in enumerate(robot_dists):
                            interp = mtri.CubicTriInterpolator(triang, data, kind='geom') # interpolate to grid
                            zi_interp = interp(*self.xy_grid)
                            zi_interp.data[zi_interp.mask] = np.mean(zi_interp) # mask handling
                            zi_interp = gaussian_filter(zi_interp,sigma=1) # smooth out
                            # zi.append(zi_interp)
                            _min, _max = np.amin(zi_interp), np.amax(zi_interp)
                            if not self.use_imshow:
                                self.robot_figs.append(self.ax1[1,idx].contourf(*self.xy_grid, zi_interp, vmin=_min, vmax=_max, levels=30, cmap=self.cmap))
                            else:
                                self.robot_figs[idx].set_data(zi_interp)
                                self.robot_figs[idx].autoscale()
                    # self.robot_figs=[self.ax1[1,0].contourf(*self.xy_grid, zi[0], vmin=_min, vmax=_max, levels=30, cmap=self.cmap),
                    #                  self.ax1[1,1].contourf(*self.xy_grid, zi[1], vmin=_min, vmax=_max, levels=30, cmap=self.cmap)]
                else:
                    _min, _max = np.amin(robot_dists), np.amax(robot_dists)
                    self.robot_figs=[self.ax1[1,0].tricontourf( *samples.T, robot_dists[0], vmin=_min, vmax=_max, levels=30, cmap=self.cmap),
                                    self.ax1[1,1].tricontourf( *samples.T, robot_dists[1], vmin=_min, vmax=_max, levels=30, cmap=self.cmap)]

                # colorbars
                # if self.cbar is not None:
                #     for cbar in self.cbar:
                #         cbar.remove()
                # elif not self.plot_zs:
                if self.cbar is None:
                    if not(self.plot_zs):
                        for ax in self.ax1.flatten():
                            box = ax.get_position()
                            ax.set_position([box.x0*0.93, box.y0, box.width, box.height])
                    self.cbar = []
                    for idx,offset in zip([0,1],[1.05,1.01]):
                        box = self.ax1[1,idx].get_position()
                        axColor = self.fig1.add_axes([box.x0*offset + box.width, box.y0+box.height*0.05, 0.02, box.height*0.9])
                        sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=_min, vmax=_max), cmap = self.cmap)
                        cbar = self.fig1.colorbar(sm, cax = axColor, orientation="vertical")
                        cbar.set_ticks([_min, _max])
                        cbar.set_ticklabels(['Low', 'High'])
                        self.cbar.append(cbar)

        if (z_mu is not None) and (z_var is not None):
            if self.z_figs is None:
                self.z_figs = [self.ax1[0,2].bar(np.arange(len(z_mu)),z_mu,color='b'),
                               self.ax1[1,2].bar(np.arange(len(z_var)),z_var,color='b')]
            else:
                for idx,data in enumerate([z_mu,z_var]):
                    for rect, h in zip(self.z_figs[idx], data):
                        rect.set_height(h)
                    self.ax1[idx,2].relim()
                    self.ax1[idx,2].autoscale_view()
        if self.render and draw:
            # self.fig1.canvas.draw_idle()
            self.fig1.canvas.blit(self.fig1.bbox)
            self.fig1.canvas.flush_events()

    def project_samples(self,samples):
        # if self.states is None:
        #     xy = samples[:,:2]
        # elif 'r' in self.states:
        #     num_samps = samples.shape[0]
        #     pos = np.zeros((num_samps,3))
        #     rot = np.zeros((num_samps,3))
        #     for idx,(vals,key) in enumerate(zip(samples.T,self.states)):
        #         if key == 'x':
        #             pos[:,0] = vals
        #         elif key == 'y':
        #             pos[:,1] = vals
        #         elif key == 'z':
        #             pos[:,2] = vals
        #         elif key == 'r':
        #             if self.convert_angle:
        #                 vals = ws_conversion(vals,self.robot_lim[[idx]],self.tray_lim[[idx]])
        #             rot[:,0] = vals
        #         elif key == 'p':
        #             if self.convert_angle:
        #                 vals = ws_conversion(vals,self.robot_lim[[idx]],self.tray_lim[[idx]])
        #             rot[:,1] = vals
        #         elif key == 'w':
        #             if self.convert_angle:
        #                 vals = ws_conversion(vals,self.robot_lim[[idx]],self.tray_lim[[idx]])
        #             rot[:,2] = vals
        #     # define sensor dirction
        #     upVec = np.array([0,0,1])
        #     r  = Rotation.from_euler('xyz',rot).as_matrix()
        #     camDir = (r @ upVec[:,None]).squeeze(-1) # get direction of sensor z-axis
        #     with np.errstate(divide='ignore'):
        #         scale = np.where(np.abs(camDir[:,-1]) > 1e-5, pos[:,-1] / camDir[:,-1], 0.)  # get z scaling to height of sensor
        #     xy = (pos - scale[:,None] * camDir)[:,:2]
        # else:
        xy = samples[:,self.plot_idx]
        return xy

    def save(self,fname,full_path=True):
        if (self.robot_figs is not None) and (self.fig1 is not None):
            if not self.use_imshow:
                for cont in self.robot_figs:
                    for c in cont.collections:
                        c.set_edgecolor("face")
                        c.set_rasterized(True)
            if not full_path:
                fname = self.path+fname+'.svg'
            self.fig1.savefig(fname,dpi=100)

from mpl_toolkits.mplot3d import Axes3D
class Plotter3D(object):
    def __init__(self,plot_idx,xinit,render=True,path=None,plot_zs=True,states=None,robot_lim=None,tray_lim=None,save_folder='explr/',tdist=False):
        self.plot_idx = plot_idx 
        lim = robot_lim[plot_idx]

        # extra to deal with angles
        self.robot_lim = robot_lim
        self.tray_lim = tray_lim
        self.states = states
        self.interp = True
        self.use_imshow = False
        self.scatter = False
        if self.interp:
            num_samples = 50
            self.update_samples(lim,num_samples)
        self.plot_future_states = True

        # general
        self.render=render
        self.path = path + save_folder
        if os.path.exists(path+save_folder) == False:
            os.makedirs(path+save_folder)

        # set up plotting (fig 1)
        self.plot_zs = plot_zs
        if plot_zs:
            figsize=(9,8)
            fig_num = 331
        else:
            figsize=(6,8)
            fig_num = 321
        fig = plt.figure(figsize=figsize)
        fig.suptitle(f"full min: {np.amin(1.):0.4f}, smoothed min: {np.amin(1.):0.4f}")
        ax = []
        ## ax[0,0]
        sensor_ax = fig.add_subplot(fig_num)
        sensor_ax.set_ylabel('Sensor View',size='large')
        sensor_ax.set_aspect('equal', adjustable='box')
        ax.append(sensor_ax)
        fig_num += 1
        ## ax[0,1]
        path_ax = fig.add_subplot(fig_num)
        path_ax.set_title('Sensor Path')
        self.poses = []
        self.pose_history = path_ax.plot(xinit[0],xinit[1],"k.",clip_on=False,zorder=0)
        path_ax.plot(xinit[0],xinit[1],"r*",clip_on=False,zorder=1)
        self.current_pose = path_ax.plot(xinit[0],xinit[1],"gs",ms=12,clip_on=False,zorder=2)
        if self.plot_future_states:
            self.pose_plan = path_ax.plot(xinit[0],xinit[1],"b.",clip_on=False,zorder=3)
        path_ax.set_aspect('equal', adjustable='box')
        ax.append(path_ax)
        fig_num += 1
        ## ax[0,2]
        if plot_zs:
            zmu_ax = fig.add_subplot(fig_num)
            zmu_ax.set_title(r'Latent Space $(\mu)$')
            ax.append(zmu_ax)
            fig_num += 1
        ## ax[1,0]
        targ_dist_ax = fig.add_subplot(fig_num)
        targ_dist_ax.set_title('Target Distribution' if tdist else 'CVAE Conditional Entropy')
        targ_dist_ax.set_aspect('equal', adjustable='box')
        ax.append(targ_dist_ax)
        fig_num += 1
        ## ax[1,1]
        traj_dist_ax = fig.add_subplot(fig_num)
        traj_dist_ax.set_title('Time Averaged Distribution') # (q)')
        traj_dist_ax.set_aspect('equal', adjustable='box')
        ax.append(traj_dist_ax)
        fig_num += 1
        ## ax[1,2]
        if plot_zs:
            zvar_ax = fig.add_subplot(fig_num)
            zvar_ax.set_title(r'Latent Space $(\sigma^2)$')
            ax.append(zvar_ax)
            fig_num += 1
        ## ax[2,0]
        targ_dist_ax2 = fig.add_subplot(fig_num, projection='3d')
        targ_dist_ax2.view_init(azim=-110)
        ax.append(targ_dist_ax2)
        fig_num += 1
        ## ax[2,1]
        traj_dist_ax2 = fig.add_subplot(fig_num, projection='3d')
        traj_dist_ax2.view_init(azim=-110)
        ax.append(traj_dist_ax2)
        fig_num += 1
        if plot_zs:
            extra_ax = fig.add_subplot(fig_num)
            ax.append(extra_ax)
            fig_num += 1
        for idx,axs in enumerate(ax):
            if idx == 0:
                # axs.axes.xaxis.set_visible(False)
                # axs.axes.yaxis.set_visible(False)
                axs.xaxis.set_ticklabels([])
                axs.xaxis.set_ticks([])
                axs.yaxis.set_ticklabels([])
                axs.yaxis.set_ticks([])
            elif plot_zs and (idx in [2,5]):
                axs.set_ylabel('.\n.\n.')
                axs.yaxis.label.set_color('white')
            elif plot_zs and (idx == 8):
                axs.axis('off')  
            else:
                axs.set_xlim(lim[0,0],lim[0,1])
                axs.set_ylim(lim[1,0],lim[1,1])
                axs.axes.xaxis.set_ticks(np.linspace(lim[0,0],lim[0,1],3))
                axs.axes.yaxis.set_ticks(np.linspace(lim[1,0],lim[1,1],3))
                plot_label = [self.states[p] if self.states[p] == self.states[p].lower() else 'd' + self.states[p].lower() + '\dt' for p in self.plot_idx]
                axs.set_xlabel(plot_label[0])
                axs.set_ylabel(plot_label[1])
        fig.tight_layout()
        if self.render:
            move_figure(fig,'topRight')
            plt.ion()
            plt.show(block=False)
        fig.canvas.blit(fig.bbox)
        self.fig1 = fig
        if plot_zs:
            self.ax1 = np.array(ax).reshape(3,3)
        else:
            self.ax1 = np.array(ax).reshape(3,2)
        self.update_plot_dims = (4.,2.)

        self.cam_view = None
        self.z_figs = None
        self.cbar = None
        self.cmap = 'RdYlBu' # sc.ScicoSequential('heat_r').get_mpl_color_map()
        mpl.style.use('fast')

        if self.use_imshow and self.interp:
            self.robot_figs = [self.ax1[1,0].imshow(np.zeros((num_samples,num_samples)),cmap=self.cmap,extent=(*lim[0],*lim[1]),origin='lower'),
                               self.ax1[1,1].imshow(np.zeros((num_samples,num_samples)),cmap=self.cmap,extent=(*lim[0],*lim[1]),origin='lower')]
        else:
            self.robot_figs = None
        self.robot_figs_3d = None

    def update_samples(self,lim,num_samples=50):
        self.xy_grid = np.meshgrid(*np.linspace(*lim.T, num_samples).T)
        # self.xy_grid = np.meshgrid(*[np.linspace(*lim, num_samples)]*2)
        self.robot_figs = None

    def update(self,args,draw=True,throttle=False):
        [cam_data,state,force,robot_plot_data,z_mu,z_var,img_pred,iter_step]= args
          
        if self.cam_view is None:
            self.cam_view = self.ax1[0,0].imshow(cam_data,cmap='gray')
        else:
            self.cam_view.set_data(cam_data)
            self.cam_view.autoscale()
        if iter_step is not None:
            self.ax1[0,0].set_title(f'explore iteration: {iter_step[0]}')
            self.ax1[0,0].set_xlabel(f'learn iteration: {iter_step[1]}',size='large')
        if (self.states is not None) and ('z' in self.states):
            marker_size = 15.+(1+state[self.states.rfind('z')]).copy()*10. # resize based on z
        else:
            marker_size = 15.
        self.poses.append(state[:2])
        self.pose_history[0].set_data(*np.array(self.poses).T)
        self.current_pose[0].set_data(state[0],state[1])
        if self.states is not None:
            self.current_pose[0].set_markersize(marker_size)
            self.current_pose[0].set_markeredgecolor('k')
            self.current_pose[0].set_markeredgewidth(min(force,5.))
        if robot_plot_data is not None:
            if self.plot_future_states:
                self.pose_plan[0].set_data(robot_plot_data[3][:,self.plot_idx[0]],robot_plot_data[3][:,self.plot_idx[1]])
            self.fig1.suptitle(f"full min: {np.amin(robot_plot_data[1]):0.4f}, smoothed min: {np.amin(robot_plot_data[4]):0.4f}")
            if not throttle: 
                ### 2d
                if not self.use_imshow:
                    if self.robot_figs is not None:
                        for cont in self.robot_figs:
                            for c in cont.collections:
                                c.remove()  # removes only the contours, leaves the rest intact
                    self.robot_figs = []

                samples = self.project_samples(robot_plot_data[0])
                robot_dists = robot_plot_data[4:6]
                if self.interp:
                    # try interpolating samples befor passing to plotting function
                    with np.errstate(invalid='ignore'):
                        triang = mtri.Triangulation(*samples.T)
                        # zi = []
                        for idx,data in enumerate(robot_dists):
                            interp = mtri.CubicTriInterpolator(triang, data, kind='geom') # interpolate to grid
                            zi_interp = interp(*self.xy_grid)
                            zi_interp.data[zi_interp.mask] = np.mean(zi_interp) # mask handling
                            zi_interp = gaussian_filter(zi_interp,sigma=1) # smooth out
                            # zi.append(zi_interp)
                            _min, _max = np.amin(zi_interp), np.amax(zi_interp)
                            if not self.use_imshow:
                                self.robot_figs.append(self.ax1[1,idx].contourf(*self.xy_grid, zi_interp, vmin=_min, vmax=_max, levels=30, cmap=self.cmap))
                            else:
                                self.robot_figs[idx].set_data(zi_interp)
                                self.robot_figs[idx].autoscale()
                    # self.robot_figs=[self.ax1[1,0].contourf(*self.xy_grid, zi[0], vmin=_min, vmax=_max, levels=30, cmap=self.cmap),
                    #                  self.ax1[1,1].contourf(*self.xy_grid, zi[1], vmin=_min, vmax=_max, levels=30, cmap=self.cmap)]
                else:
                    _min, _max = np.amin(robot_dists), np.amax(robot_dists)
                    self.robot_figs=[self.ax1[1,0].tricontourf( *samples.T, robot_dists[0], vmin=_min, vmax=_max, levels=30, cmap=self.cmap),
                                    self.ax1[1,1].tricontourf( *samples.T, robot_dists[1], vmin=_min, vmax=_max, levels=30, cmap=self.cmap)]

                ### 3d
                robot_dists = robot_plot_data[1:3]
                _min, _max = np.amin(robot_dists), np.amax(robot_dists)
                samples = robot_plot_data[0][:,:3]
                if self.robot_figs_3d is None:
                    self.robot_figs_3d=[ax.scatter( *samples.T, c=data, alpha=data, vmin=_min, vmax=_max, cmap=self.cmap,depthshade=False) for ax,data in zip([self.ax1[2,0,],self.ax1[2,1]],robot_dists)]
                else: 
                    for ax,data in zip(self.robot_figs_3d,robot_dists):
                        ax.set(clim=(_min,_max),alpha=data,array=data,offsets=samples[:,:2]) # array=color, offset=xy
                        ax.set_3d_properties(samples[:,2],'z') 
                        ax.autoscale()
                # colorbars
                # if self.cbar is not None:
                #     for cbar in self.cbar:
                #         cbar.remove()
                # elif not self.plot_zs:
                if self.cbar is None:
                    if not(self.plot_zs):
                        for idx, ax in enumerate(self.ax1.flatten()):
                            box = ax.get_position()
                            y_scale = 0. if idx > 5 else 1.0
                            ax.set_position([box.x0*0.93, box.y0*y_scale, box.width, box.height])

                    self.cbar = []
                    for idx,offset in zip([0,1],[1.05,1.01]):
                        box = self.ax1[1,idx].get_position()
                        axColor = self.fig1.add_axes([box.x0*offset + box.width, box.y0+box.height*0.05, 0.02, box.height*0.9])
                        sm = plt.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=_min, vmax=_max), cmap = self.cmap)
                        cbar = self.fig1.colorbar(sm, cax = axColor, orientation="vertical")
                        cbar.set_ticks([_min, _max])
                        cbar.set_ticklabels(['Low', 'High'])
                        self.cbar.append(cbar)
                        
                    for ax in self.ax1[2]:
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0+box.height*0.2, box.width, box.height])

        if (z_mu is not None) and (z_var is not None):
            if self.z_figs is None:
                self.z_figs = [self.ax1[0,2].bar(np.arange(len(z_mu)),z_mu,color='b'),
                               self.ax1[1,2].bar(np.arange(len(z_var)),z_var,color='b')]
            else:
                for idx,data in enumerate([z_mu,z_var]):
                    for rect, h in zip(self.z_figs[idx], data):
                        rect.set_height(h)
                    self.ax1[idx,2].relim()
                    self.ax1[idx,2].autoscale_view()
        if self.render and draw:
            # self.fig1.canvas.draw_idle()
            self.fig1.canvas.blit(self.fig1.bbox)
            self.fig1.canvas.flush_events()
            
    def save(self,fname,full_path=True):
        if (self.robot_figs is not None) and (self.fig1 is not None):
            if not full_path:
                fname = self.path+fname+'.svg'
            self.fig1.savefig(fname,dpi=100)

    def project_samples(self,samples):
        xy = samples[:,self.plot_idx]
        return xy


class TrainingPlotter(object):
    def __init__(self,render=True,path=None):
        # general
        self.render=render
        self.path = path
        for p in ['vae/','vae_checkpoint/']:
            if os.path.exists(path+p) == False:
                os.makedirs(path+p)

        self.update_plot_dims = (4.,2.)
        self.fig2 = None
        self.fig3 = None
        self.training_figs = None
        self.checkpoint_figs = None
        mpl.style.use('fast')

    def training_update(self,args):
        [y,y_pred_pre,y_pred_post,iter_step]=args
        y_pred_pre = np.clip(y_pred_pre,0,1)
        y_pred_post = np.clip(y_pred_post,0,1)
        if self.training_figs is None:
            # set up plotting (fig 2)
            fig,ax = plt.subplots(1,3,figsize=self.update_plot_dims)
            ax[0].set_title('input')
            for idx,axs in enumerate(ax.ravel()):
                if idx == 0:
                    axs.set_ylabel('VAE Training',size='large',weight='bold')
                    axs.yaxis.set_ticklabels([])
                    axs.yaxis.set_ticks([])
                else:
                    axs.axes.yaxis.set_visible(False)
                axs.xaxis.set_ticklabels([])
                axs.xaxis.set_ticks([])
                axs.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            if self.render:
                move_figure(fig,'bottomLeft',0,-fig.get_figheight()*fig.dpi)
                plt.ion()
                plt.show(block=False)
            self.fig2 = fig
            self.ax2 = ax
            self.training_figs = []
            for fig,data in zip(ax,[y,y_pred_pre,y_pred_post]):
                self.training_figs.append(fig.imshow(data,cmap='gray'))
        else:
            for fig,data in zip(self.training_figs,[y,y_pred_pre,y_pred_post]):
                fig.set_data(data)
                fig.autoscale()
        self.ax2[1].set_title(f'predicted\nupdate {iter_step[0]-iter_step[1]}')
        self.ax2[2].set_title(f'predicted\nupdate {iter_step[0]}')
        if self.render:
            self.fig2.canvas.draw_idle()
            self.fig2.canvas.flush_events()

    def checkpoint_update(self,args):
        [y,y_pred_pre,y_pred_post,iter_step]=args
        y_pred_pre = np.clip(y_pred_pre,0,1)
        y_pred_post = np.clip(y_pred_post,0,1)
        if self.checkpoint_figs is None:
            # set up plotting (fig 2)
            fig,ax = plt.subplots(1,3,figsize=self.update_plot_dims)
            ax[0].set_title('input')
            for idx,axs in enumerate(ax.ravel()):
                if idx == 0:
                    axs.set_ylabel('VAE Checkpoint',size='large',weight='bold')
                    axs.yaxis.set_ticklabels([])
                    axs.yaxis.set_ticks([])
                else:
                    axs.axes.yaxis.set_visible(False)
                axs.xaxis.set_ticklabels([])
                axs.xaxis.set_ticks([])
                axs.set_aspect('equal', adjustable='box')
            fig.tight_layout()
            if self.render:
                move_figure(fig,'bottomLeft')
                plt.ion()
                plt.show(block=False)
            self.fig3 = fig
            self.ax3 = ax
            self.checkpoint_figs = []
            for fig,data in zip(ax,[y,y_pred_pre,y_pred_post]):
                self.checkpoint_figs.append(fig.imshow(data,cmap='gray'))
        else:
            for fig,data in zip(self.checkpoint_figs,[y,y_pred_pre,y_pred_post]):
                fig.set_data(data)
                fig.autoscale()
        self.ax3[1].set_title(f'predicted\nupdate {iter_step[0]-iter_step[1]}')
        self.ax3[2].set_title(f'predicted\nupdate {iter_step[0]}')
        if self.render:
            self.fig3.canvas.draw_idle()
            self.fig3.canvas.flush_events()

    def save(self,fname,main_fname=None): #placeholder for old formatting
        if self.fig2 is not None:
            self.fig2.savefig(self.path+'vae/'+fname+'_vae.svg')
        if self.fig3 is not None:
            self.fig3.savefig(self.path+'vae_checkpoint/'+fname+'_vae_checkpoint.svg')

    def save_fig3_only(self,fname):
        self.fig3.savefig(self.path+'vae_checkpoint/'+fname+'_vae_checkpoint.svg')


class EvalPlotter(object):
    def __init__(self,render=True,path=None,sim=True,plot_seed=False,method='erg',save_folder='eval/',plot_zmu=True,plot_zvar=False):
        self.render=render
        self.plot_seed=plot_seed
        if os.path.exists(path+save_folder) == False:
            os.makedirs(path+save_folder)
        self.path = path + save_folder
        # set up plotting (fig 1)
        titles = []
        if self.plot_seed:
            titles = titles + ['Seed Image']
        titles = titles + ['Actual Image']
        self.plot_zmu = plot_zmu
        self.plot_zvar = plot_zvar
        if self.plot_zmu and self.plot_zvar: # both
            titles = titles + [r'Latent Space $(\mu)$']
            titles = titles + [r'Latent Space $(\sigma^2)$']
        elif self.plot_zmu or self.plot_zvar: # only one
            titles = titles + ['Latent Space']
        titles = titles + ['Imagined Image'] #['Entropy'] #r'Entropy (KL-E$^2$)']
        num_subplots = len(titles)
        fig,axs = plt.subplots(1,num_subplots,figsize=(2.5*(num_subplots),2.5))
        for idx,(ax,title) in enumerate(zip(axs.flatten(),titles)):
            ax.set_title(title)
            if 'Image' in title:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        if self.render:
            plt.ion()
            plt.show(block=False)
        self.img = []
        self.ax = axs

        self.fig = fig
        self.axs = axs
        self.seed = None
        self.y = None
        self.y_pred = None
        self.latent_space = None
        mpl.style.use('fast')

    def set_fingerprint_label(self,name):
        self.ax[0].axes.yaxis.set_visible(True)
        self.ax[0].set_ylabel(name,size='large')
        self.ax[0].yaxis.set_ticklabels([])
        self.ax[0].yaxis.set_ticks([])
        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def update(self,seed,y,y_pred,latent_space):
        start_idx = 0
        if self.plot_seed:
            start_idx = 1
            if self.seed is None:
                self.seed = self.ax[0].imshow(seed,cmap='gray')
            else:
                self.seed.set_data(seed)
                self.seed.autoscale()
        if self.y is None:
            self.y = self.ax[start_idx].imshow(y,cmap='gray')
        else:
            self.y.set_data(y)
            self.y.autoscale()
        if self.latent_space is not None:
            for bar in self.latent_space:
                bar.remove()
        if latent_space is not None:
            if self.plot_zmu and self.plot_zvar: # both
                self.latent_space = []
                for z_idx,z in enumerate(latent_space):
                    self.latent_space.append(self.ax[start_idx+1+z_idx].bar(np.arange(len(z)),z,color='b') )
                    self.ax[start_idx+1+z_idx].relim()
                    self.ax[start_idx+1+z_idx].autoscale_view()
                start_idx += len(latent_space)
            elif self.plot_zmu or self.plot_zvar: # only one
                self.latent_space = []
                z = latent_space[0] if self.plot_zmu else latent_space[1]
                self.latent_space.append(self.ax[start_idx+1].bar(np.arange(len(z)),z,color='b') )
                self.ax[start_idx+1].set_ylim(-2,2)
                # self.ax[start_idx+1].relim()
                # self.ax[start_idx+1].autoscale_view()
                start_idx += 1
        if self.y_pred is None:
            self.y_pred = self.ax[start_idx+1].imshow(y_pred,cmap='gray')
        else:
            self.y_pred.set_data(y_pred)
            self.y_pred.autoscale()
        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self,fname):
        self.fig.savefig(self.path+fname+'.svg')

    def close(self):
        plt.close(self.fig)

class FPEvalPlotter(object):
    def __init__(self,render=True,path=None,plot_seed=False,save_folder='eval/',suptitle=''):
        self.render=render
        self.plot_seed=plot_seed
        if os.path.exists(path+save_folder) == False:
            os.makedirs(path+save_folder)
        self.path = path + save_folder
        # set up plotting (fig 1)
        titles = []
        titles = titles + ['Stored Image']
        titles = titles + [r'$z_{stored}$']
        titles = titles + ['Test Image']
        titles = titles + [r'$z_{test}$']
        titles = titles + ['Predicted Image'] #['Entropy'] #r'Entropy (KL-E$^2$)']
        titles = titles + [r'$z_{pred}$']
        fig,axs = plt.subplots(3,2,figsize=(2.5*(2),2.5*3))
        for idx,(ax,title) in enumerate(zip(axs.flatten(),titles)):
            ax.set_title(title)
            if 'Image' in title:
                ax.axes.xaxis.set_visible(False)
                ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        plt.suptitle(suptitle)
        if self.render:
            plt.ion()
            plt.show(block=False)
        self.img = []
        self.ax = axs

        self.fig = fig
        self.axs = axs
        self.y_stored = None
        self.y_test = None
        self.y_pred = None
        self.latent_space = None
        mpl.style.use('fast')

    def update(self,y_stored,y_test,y_pred,latent_space):
        if self.y_stored is None:
            self.y_stored = self.ax[0][0].imshow(y_stored,cmap='gray')
        else:
            self.y_stored.set_data(y_stored)
            self.y_stored.autoscale()
        if self.y_test is None:
            self.y_test = self.ax[1][0].imshow(y_test,cmap='gray')
        else:
            self.y_test.set_data(y_test)
            self.y_test.autoscale()
        if self.y_pred is None:
            self.y_pred = self.ax[2][1].imshow(y_pred,cmap='gray')
        else:
            self.y_pred.set_data(y_pred)
            self.y_pred.autoscale()
        if self.latent_space is not None:
            for bar in self.latent_space:
                bar.remove()
        if latent_space is not None:
            self.latent_space = []
            for z_idx,z in enumerate(latent_space):
                self.latent_space.append(self.ax[1][z_idx].bar(np.arange(len(z)),z,color='b') )
                self.ax[1][z_idx].relim()
                self.ax[1][z_idx].autoscale_view()
            start_idx += len(latent_space)
        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self,fname):
        self.fig.savefig(self.path+fname+'.svg')


class MultiEvalPlotter(object):
    def __init__(self,render=True,path=None,data_order=[],reshape=None):
        self.render=render
        self.reshape=reshape
        if os.path.exists(path) == False:
            os.makedirs(path)
        self.path = path
        # set up titles
        titles =['Seed Image','Actual Image']
        for method,pybullet in data_order:
            if pybullet == True:
                source = "Simulation"
            else:
                source = "Hardware"
            if method == 'uniform':
                explr_method = "Random"
            else:
                explr_method = "Active"
            titles.append(f'{source} | {explr_method}')
        num_subplots = len(titles)
        fig,axs = plt.subplots(1,num_subplots,figsize=(2.5*(num_subplots),3))
        for ax,title in zip(axs.flatten(),titles):
            ax.set_title(title)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        plt.tight_layout()
        if self.render:
            plt.ion()
            plt.show(block=False)
        self.img = []
        self.ax = axs

        self.fig = fig
        self.axs = axs
        self.seed = None
        self.y = None
        self.y_pred = [None]*len(data_order)
        mpl.style.use('fast')

    def update(self,seed,y,y_preds):
        if self.seed is None:
            self.seed = self.ax[0].imshow(seed,cmap='gray')
        else:
            self.seed.set_data(seed)
            self.seed.autoscale()
        y = self.reshape(y)
        if self.y is None:
            self.y = self.ax[1].imshow(y,cmap='gray')
        else:
            self.y.set_data(y)
            self.y.autoscale()
        # each method
        if self.y_pred[0] is None:
            for idx,y_pred in enumerate(y_preds):
                self.y_pred[idx] = self.ax[idx+2].imshow(self.reshape(y_pred.squeeze()),cmap='gray')
        else:
            for idx,y_pred in enumerate(y_preds):
                self.y_pred[idx].set_data(self.reshape(y_pred.squeeze()))
                self.y_pred[idx].autoscale()

        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self,fname):
        self.fig.savefig(self.path+fname+'.svg')

class DebugPlotter(object):
    def __init__(self,render=True,path=None):
        self.render=render
        if os.path.exists(path) == False:
            os.makedirs(path)
        self.path = path

        num_samples = 10
        fig,axs = plt.subplots(2,num_samples,figsize=(1.5*(num_samples),1.5*(2)))
        for ax in axs.flatten():
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
        axs[0,0].set_ylabel('Actual')
        axs[1,0].set_ylabel('Imagined')
        plt.tight_layout()
        if self.render:
            plt.ion()
            plt.show(block=False)
        self.img = []
        self.ax = axs

        self.fig = fig
        self.axs = axs
        self.y = [None]*num_samples
        self.y_pred = [None]*num_samples
        mpl.style.use('fast')

    def update(self,ys,y_preds,seed=None,y_preds_seeded=None):
        ys=np.clip(ys,0.,1.)
        y_preds=np.clip(y_preds,0.,1.)
        # each sample
        if self.y[0] is None:
            for idx,(y,y_pred) in enumerate(zip(ys,y_preds)):
                self.y[idx] = self.ax[0,idx].imshow(y.squeeze(),cmap='gray')
                self.y_pred[idx] = self.ax[1,idx].imshow(y_pred.squeeze(),cmap='gray')
        else:
            for idx,(y,y_pred) in enumerate(zip(ys,y_preds)):
                self.y[idx].set_data(y.squeeze())
                self.y[idx].autoscale()
                self.y_pred[idx].set_data(y_pred.squeeze())
                self.y_pred[idx].autoscale()

        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def draw(self): 
        if self.render:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def save(self,fname):
        self.fig.savefig(self.path+fname+'.svg')
