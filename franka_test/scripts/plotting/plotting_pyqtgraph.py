#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import os
from scipy.ndimage import gaussian_filter
import matplotlib.tri as mtri
from scipy.interpolate import griddata

from scipy.spatial.transform import Rotation
from franka.franka_utils import ws_conversion

import pyqtgraph as pg
from pyqtgraph.exporters import SVGExporter,ImageExporter
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

pg.setConfigOption('background', 'w')
# pg.setConfigOption('background', (255,255,255, 10)) # transparent for export
pg.setConfigOption('foreground', 'k')

use_smoothed_dists = True

from math import log10, floor

def find_exp(number) -> int:
    base10 = log10(abs(number))
    return abs(floor(base10))

class Plotter(object):
    def __init__(self,plot_idx,xinit,render=True,path=None,plot_zs=True,states=None,robot_lim=None,tray_lim=None,save_folder='explr/',tdist=False,axis_equal=False):
        # extra to deal with angles
        self.plot_idx = plot_idx 
        lim = robot_lim[plot_idx]
        ws_scale = 1.15
        self.lim = np.array(lim)
        self.lim += np.tile(np.array([[-1.,1.]]),(len(self.lim),1))*(self.lim[:,[1]]-self.lim[:,[0]])*(ws_scale-1.)/2.
        size = self.lim[:,1]-self.lim[:,0]
        if axis_equal:
            pad_axes = np.tile(np.array([[-1.,1.]]),(len(self.lim),1))*(np.max(size)-size)[:,None]/2.
        else: 
            pad_axes = 0.
        self.xaxis_lim,self.yaxis_lim = self.lim + pad_axes
        self.robot_lim = robot_lim
        self.tray_lim = tray_lim
        self.convert_angle = not(np.any(robot_lim == None)) and not(np.any(tray_lim == None))
        self.states = states
        self.plot_samples = False
        self.plot_future_states = True
        self.use_svg = False
        self.update_samples(self.lim)

        # general
        self.plot_zs = plot_zs
        self.render=render
        self.path = path + save_folder
        if os.path.exists(path+save_folder) == False:
            os.makedirs(path+save_folder)

        self.scale = 0.9

        # set up plotting (fig 1)
        self.app = pg.mkQApp()
        try:
            self.app.setFont(QtGui.QFont('Nimbus Roman',12*self.scale)) # change font
        except:
            pass
        self.fig1 = pg.GraphicsLayoutWidget(show=render)
        self.fig1.setRenderHints(QtGui.QPainter.Antialiasing)
        self.fig1.setWindowTitle('Exploration')

        if self.plot_zs:
            width = 1200
            titles = [['Sensor View',                    
                    '<html>Latent Space (&mu;)</html>', 
                    '<html>Latent Space (&sigma;<sup>2</sup>)</html>',
                    'Sensor Pred'],
                    [f'Sensor Path ({states})',
                    'Time Averaged Dist.',
                    'Target Distribution' if tdist else 'Conditional Entropy',
                    'Cost']]
        else:
            width = 600
            titles = [['Sensor View',
                    f'Sensor Path ({states})','','Sensor Pred'],
                    ['Target Distribution' if tdist else 'Conditional Entropy',
                    'Time Averaged Dist.','','']]
        shift = False
        bars = 'zwbrp'
        num_bars = sum([b in states for b in bars])
        if num_bars > 1:  
            width += 50*num_bars
            shift = True
        width = int(width*self.scale)
        self.width = width/len(titles[0])
        height = int(650*self.scale)
        # self.fig1.setBaseSize(width,height)
        self.fig1.setFixedWidth(width)
        self.fig1.setFixedHeight(height)

        self.ax1 = np.ones(np.array(titles).shape,dtype='object')
        for row_idx, (title) in enumerate(titles):
            for col_idx, (t) in enumerate(title):
                plot = self.fig1.addPlot(row=row_idx, col=col_idx)
                plot.hideButtons()
                plot.setTitle(t,size= "{}pt".format(16*self.scale))
                for loc in ['right','top']:
                    plot.showAxis(loc)
                    plot.getAxis(loc).setStyle(tickLength=0)
                    plot.getAxis(loc).setTicks([[]])
                if 'Path' in t:
                    plot.setRange(xRange=[lim[0,0],lim[0,1]], yRange=[lim[1,0],lim[1,1]])
                    plot_label = [self.states[p] if self.states[p] == self.states[p].lower() else 'd' + self.states[p].lower() + '/dt' for p in self.plot_idx]
                    plot.setLabel('bottom', plot_label[0])
                    plot.setLabel('left', plot_label[1])
                    path_ax = plot
                elif 'Latent Space' in t:
                    for loc in ['left','right','bottom']:
                        plot.setLabel(loc,'<html>&nbsp;</html>')
                elif 'Sensor' in t:
                        plot.setLabel('bottom','<html>&nbsp;<br>&nbsp;</html>')
                        for loc in ['top','bottom','left','right']:
                            plot.getAxis(loc).setStyle(tickLength=0)
                            plot.getAxis(loc).setTicks([[]])
                            plot.getAxis(loc).setPen(pg.mkPen(width=0,color='white'))
                elif ('Dist' in t) or ('Ent' in t):
                    for loc in ['bottom','left']:
                        plot.setLabel(loc,'<html>&nbsp;</html>')
                elif t=='' :
                    [plot.hideAxis(loc) for loc in ['bottom','left','right','top']]
                # turn off interactive mouse
                plot.setMouseEnabled(x=False)
                plot.setMouseEnabled(y=False)
                if not(('Latent' in t) or ('Cost' in t)):
                    plot.setAspectLocked(True)
                self.ax1[row_idx,col_idx] = plot

        # if shift and self.plot_zs: 
        #     self.ax1[0,0].setLabel('right','<html>&nbsp;</html>')
        #     self.ax1[0,1].setLabel('right','<html>&nbsp;</html>')
        #     self.ax1[1,0].setLabel('right','<html>&nbsp;</html>')


        self.fig1.ci.layout.setContentsMargins(0, 0, 0, 0)
        self.fig1.ci.layout.setSpacing(0)  # might not be necessary for you

        self.poses = 1
        self.x_poses = [xinit[0]]
        self.y_poses = [xinit[1]]
        self.pose_markers = []
        for idx,(size,color,symbol) in enumerate(zip([5,5,10,15,5], ['k','k','r',(45,112,13),'b'], ['o','o','star','s','o'])):
            if idx == 0 and self.use_svg:
                scatter = pg.PlotDataItem([xinit[0]],[xinit[1]],pen=pg.mkPen(width=3,color=color),symbol=None, skipFiniteCheck=True)
            elif idx == 4:
                if self.plot_future_states:
                    scatter = pg.PlotDataItem([xinit[0]],[xinit[1]],pen=None,symbolPen=pg.mkPen(width=1, color=color),symbolBrush=pg.mkBrush(color=color), symbol=symbol,symbolSize=size,skipFiniteCheck=True)
            else:
                scatter = pg.PlotDataItem([xinit[0]],[xinit[1]],pen=None,symbolPen=pg.mkPen(width=1, color=color),symbolBrush=pg.mkBrush(color=color), symbol=symbol,symbolSize=size,skipFiniteCheck=True)
            path_ax.addItem(scatter)
            self.pose_markers.append(scatter)
        self.pose_markers[0].setAlpha(0.2,False)

        corners = np.array([[lim[0,0],lim[1,0]],
                            [lim[0,1],lim[1,0]],
                            [lim[0,1],lim[1,1]],
                            [lim[0,0],lim[1,1]],
                            [lim[0,0],lim[1,0]]])
        rect = pg.PlotDataItem(corners,pen=pg.mkPen(width=3,color='k'), skipFiniteCheck=True)
        rect.setAlpha(0.5,False)
        path_ax.addItem(rect)
        self.format_path_plot()

        self.extra_ax = None
        plot_states = [states[idx] for idx in plot_idx]
        if len(self.states) > 2: 
            for bar in bars: 
                if (bar in self.states) and not(bar in plot_states):
                    self.build_extra(xinit,bar,path_ax)

        self.text_scale=None
        self.cam_view = None
        self.pred_view = None
        self.robot_figs = None
        self.z_figs = None
        self.cbar = None
        self.cost = None
        self.cost_val = None
        self.cost_markers = None
        self.text_cost = None
        self.cmap = pg.colormap.getFromMatplotlib('gist_heat')
        if self.use_svg:
            self.exporter = SVGExporter(self.fig1.scene())
        else:
            self.exporter = ImageExporter(self.fig1.scene())

        plot_loc = QtGui.QGuiApplication.primaryScreen().availableGeometry().topRight()
        plot_loc.setX( plot_loc.x() - width)
        self.fig1.move(plot_loc)
        # self.fig1.showFullScreen()
        self.app.processEvents()

    def update_samples(self,lim):
        num_samples = 100
        # self.xy_grid = np.meshgrid(*[np.linspace(*self.lim, num_samples)]*2,indexing='ij')
        self.xy_grid = np.meshgrid(*np.linspace(*lim.T, num_samples).T,indexing='ij')
        if 'r' in self.states:
            self.grid_samples = self.xy_grid
        else:
            self.grid_samples = self.xy_grid + [np.zeros((num_samples,num_samples))]*(len(self.states)-2)

    def format_path_plot(self):
        self.ax1[1,0].setRange(xRange=self.xaxis_lim, yRange=self.yaxis_lim, padding=0)
        # for loc in ['bottom','left']:
            # self.ax1[1,0].getAxis(loc).setTickSpacing(1.0,0.1)

    def build_cam_view(self,cam_data):
        self.cam_view = pg.ImageItem(cam_data,levels=(0.,1.))
        self.ax1[0,0].addItem(self.cam_view)
        lim = [0,cam_data.shape[0]]
        self.ax1[0,0].setRange(xRange=lim, yRange=lim, padding=0)
        self.app.processEvents()

    def build_pred_view(self,pred_data):
        self.pred_data = pg.ImageItem(pred_data,levels=(0.,1.))
        self.ax1[0,3].addItem(self.pred_data)
        lim = [0,pred_data.shape[0]]
        self.ax1[0,3].setRange(xRange=lim, yRange=lim, padding=0)
        self.app.processEvents()

    def build_cost(self,cost_data,val):
        self.cost = [cost_data]
        self.cost_val = [val]
        self.cost_markers = pg.PlotDataItem(self.cost_val,self.cost,pen=pg.mkPen(width=3,color='k'),symbol=None, skipFiniteCheck=True)
        self.ax1[1,3].addItem(self.cost_markers)
        labelStyle = {'font-size': '{}pt'.format(12*self.scale)}
        self.ax1[1,3].setLabel("bottom",'exploration steps',**labelStyle)

        self.text_cost = pg.LabelItem('')
        self.text_cost.setParentItem(self.ax1[1,3])
        self.text_cost.anchor(itemPos=(1,0), parentPos=(1,0), offset=(-20,40))

    def build_scatter(self,points):
        color = (0,255,0,100)
        self.scatter = pg.PlotDataItem(points,pen=None,symbolPen=pg.mkPen(width=1,color=color),symbolBrush=pg.mkBrush(color=color), symbol='o',symbolSize=2,skipFiniteCheck=True)
        offset = 1 if self.plot_zs else 0
        self.ax1[1,offset].addItem(self.scatter)
        self.app.processEvents()

    def build_extra(self,x,axis,path_ax): 
        if self.extra_ax is None: 
            self.extra_ax = []
            self.extra_idx = []
            self.extra_axis_lim = []
            self.extra_poses = []
            self.extra_pose_markers = []
        extra_idx = self.states.rfind(axis)
        extra_lim = self.robot_lim[extra_idx]

        extra_ax = pg.PlotItem()
        for loc,label in zip(['right','left','top','bottom'],['','',axis,'']):
            extra_ax.showAxis(loc)
            extra_ax.getAxis(loc).setStyle(tickLength=0)
            extra_ax.getAxis(loc).setTicks([[]])
            extra_ax.getAxis(loc).setLabel((label))
        buff = (extra_lim[1]-extra_lim[0])/2
        extra_axis_lim = [extra_lim[0]-buff ,extra_lim[1]+buff]
        extra_ax.getAxis('right').setRange(*extra_axis_lim )

        extra_pose_markers = []
        for idx,(size,color,symbol) in enumerate(zip([3,5,10,15,5], ['k','k','r',(45,112,13),'b'], [None,'o','star','s','o'])):
            if idx == 0:
                scatter = pg.PlotDataItem([0],[x[extra_idx]],pen=pg.mkPen(width=size,color=color),symbol=symbol, skipFiniteCheck=True)
            elif idx == 4:
                if self.plot_future_states:
                    scatter = pg.PlotDataItem([0],[x[extra_idx]],pen=None,symbolPen=pg.mkPen(width=1,    color=color),symbolBrush=pg.mkBrush(color=color), symbol=symbol,symbolSize=size,skipFiniteCheck=True)
            else:
                scatter = pg.PlotDataItem([0],[x[extra_idx]],pen=None,symbolPen=pg.mkPen(width=1,    color=color),symbolBrush=pg.mkBrush(color=color), symbol=symbol,symbolSize=size,skipFiniteCheck=True)
            extra_ax.addItem(scatter)
            extra_pose_markers.append(scatter)
        extra_pose_markers[0].setAlpha(0.2,False)

        lim = np.array([[-1,1], extra_lim.tolist()])
        corners = np.array([[lim[0,0],lim[1,0]],
                    [lim[0,1],lim[1,0]],
                    [lim[0,1],lim[1,1]],
                    [lim[0,0],lim[1,1]],
                    [lim[0,0],lim[1,0]]])
        rect = pg.PlotDataItem(corners,pen=pg.mkPen(width=3,color='k'), skipFiniteCheck=True)
        rect.setAlpha(0.5,False)
        extra_ax.addItem(rect)

        count = len(self.extra_ax)+2
        path_ax.layout.addItem( extra_ax, 2, count) # insert in layout after right-hand axis
        ## fix spacing
        path_ax.layout.setColumnFixedWidth(0, 25-count) 
        for c in range(2,count+1):
            path_ax.layout.setColumnFixedWidth(c, 1)

        self.extra_ax.append(extra_ax)
        self.extra_idx.append(extra_idx)
        self.extra_axis_lim.append(extra_axis_lim)
        self.extra_pose_markers.append(extra_pose_markers)
        self.extra_poses.append([x[extra_idx]])
        self.app.processEvents()

    def build_robot_figs(self,robot_data):
        self.robot_figs = []
        self.cbar = []
        lut = self.cmap.getLookupTable(0.0, 1.0)
        robot_rect = [self.lim[0,0],self.lim[1,0]] + [self.lim[0,1]-self.lim[0,0],self.lim[1,1]-self.lim[1,0]]
        offset = 1 if self.plot_zs else 0
        for idx,data in enumerate(robot_data):
            ## make heatmap
            img = pg.ImageItem(data,lut=lut,autoRange=False,autoLevels=True,rect=robot_rect)
            self.ax1[1,idx+offset].addItem(img)
            self.ax1[1,idx+offset].setRange(xRange=self.xaxis_lim, yRange=self.yaxis_lim, padding=0)
            self.robot_figs.append(img)
            ## contours
            # curves = []
            # levels = np.linspace(data.min(), data.max(), 5)
            # for i in range(len(levels)):
                # v = levels[i]
                # generate isocurve with automatic color selection
                # c = pg.IsocurveItem(level=v, pen=(i, len(levels)*1.5))
                # c.setParentItem(img)  ## make sure isocurve is always correctly displayed over image
                # c.setZValue(10)
                # curves.append(c)
            # self.robot_figs.append([img,curves])
        idx = 1 + offset
        # make colorbar
        colorbar = pg.ColorBarItem(interactive=False,colorMap=self.cmap,width=int(20*self.scale))
        colorbar.setFixedWidth(int(20*self.scale))
        colorbar.setImageItem(img,insert_in=self.ax1[1,idx])
        colorbar.getAxis('right').setRange(-0.05,1.05)
        colorbar.getAxis('top').setLabel('High')
        colorbar.getAxis('bottom').setLabel('Low')
        colorbar.getAxis('right').setTicks([])
        colorbar.getAxis('right').setStyle(tickLength=0)
        # store for future use
        self.cbar.append(colorbar)
        self.app.processEvents()

    def build_bar_graphs(self,z_vals):
        self.z_figs = []
        for idx,data in enumerate(z_vals):
            self.z_figs.append(pg.BarGraphItem(x=range(len(data)), height=data, width=0.7, pen=(0,0,0,0), brush='b'))
            self.ax1[0,idx+1].addItem(self.z_figs[idx])
            # self.ax1[0,idx+1].setFixedWidth(self.width)
        self.app.processEvents()

    def update(self,args,proj_state=True,draw=True,throttle=False):
        [cam_data,state,force,robot_plot_data,z_mu,z_var,img_pred,iter_step]= args
        ## rotate in gui
        # cam_data = np.rot90(cam_data,axes=(1,0))
        # img_pred = np.rot90(img_pred,axes=(1,0))
        if not throttle: 
            if self.cam_view is None:
                self.build_cam_view(cam_data)
            else:
                self.cam_view.setImage(cam_data,autoLevels=False)
            if iter_step is not None:
                labelStyle = {'font-size': '{}pt'.format(14*self.scale)}
                if self.plot_zs:
                    self.ax1[0,0].setLabel('left',f'update iteration: {iter_step[1]}',**labelStyle)
                else:
                    self.ax1[0,0].setLabel('left',f'explore iteration: {iter_step[0]}',**labelStyle)
                    self.ax1[1,0].setLabel('left',f'update iteration: {iter_step[1]}',**labelStyle)
        if (self.states is not None) and ('z' in self.states):
            z = state[self.states.rfind('z')]
            marker_size = 25.+(1+z).copy()*10. # resize based on z
        else:
            marker_size = 25.
        marker_outline=pg.mkPen(width=min(force,5.),color='k')
        if self.extra_ax is not None: 
            for idx,extra_ax in enumerate(self.extra_ax):
                val = state[self.extra_idx[idx]]
                self.extra_poses[idx].append(val)

                extra_poses = self.extra_poses[idx]
                extra_pose_markers = self.extra_pose_markers[idx]
                extra_axis_lim = self.extra_axis_lim[idx]
                extra_pose_markers[0].setData(x=np.zeros_like(extra_poses[:-20]),y=extra_poses[:-20])
                extra_pose_markers[1].setData(x=np.zeros_like(extra_poses[-20:]),y=extra_poses[-20:])
                extra_pose_markers[3].setData([0],[val],size=marker_size,symbolPen=marker_outline)
                extra_ax.setRange(xRange = [-0.1,0.1], yRange=extra_axis_lim, padding=0)
        if proj_state:
            state = self.project_samples(state[None,:]).squeeze()
        self.poses += 1
        self.x_poses.append(state[0])
        self.y_poses.append(state[1])
        self.pose_markers[0].setData(x=self.x_poses[:-20],y=self.y_poses[:-20])
        self.pose_markers[1].setData(x=self.x_poses[-20:],y=self.y_poses[-20:])
        self.pose_markers[3].setData([state[0]],[state[1]],size=marker_size,symbolPen=marker_outline)
        # try:
        if robot_plot_data is not None:
            ## cost
            if self.cost is None:
                self.build_cost(robot_plot_data[-1].item(),iter_step[0])
            else:
                self.cost.append(robot_plot_data[-1].item())
                self.cost_val.append(iter_step[0])
                self.cost_markers.setData(x=self.cost_val,y=self.cost)
            cost_str = ''
            for val,prefix in zip([min(self.cost[1:]) if len(self.cost) > 1 else self.cost[0],self.cost[-1]],
                                  ['min: ',', new:']):
                if val > 10: 
                    cost_str = cost_str + prefix + f'{val:0.0f}'
                else:
                    cost_str = cost_str + prefix + f'{val:0.2f}'
            self.text_cost.setText(cost_str)

            if not throttle: 
                ## reformat 
                samples = self.project_samples(robot_plot_data[0])
                if use_smoothed_dists: 
                    robot_dists = robot_plot_data[4:6]
                else:
                    robot_dists = robot_plot_data[1:3]
                if self.plot_zs:
                    robot_dists = robot_dists[::-1]
                labelStyle = {'font-size': '{}pt'.format(12*self.scale)}
                if len(self.states)>2: 
                    targ_str = f"full min: {np.amin(robot_plot_data[1]):0.4f}, smoothed min:{np.amin(robot_plot_data[4]):0.4f}" 
                    targ_str2 = f"full min: {np.amin(robot_plot_data[2]):0.4f}, smoothed min:{np.amin(robot_plot_data[5]):0.4f}"
                else: 
                    targ_str = f"min: {np.amin(robot_dists[0]):0.4f}"
                if self.plot_zs:
                    self.ax1[1,2].setLabel("bottom",targ_str,**labelStyle)
                    self.ax1[1,1].setLabel("bottom",targ_str2,**labelStyle)
                else:
                    self.ax1[1,0].setLabel("bottom",targ_str,**labelStyle)
                # try interpolating samples befor passing to plotting function
                #### scipy interpolation
                fill = np.mean(robot_dists)
                # robot_data = [griddata(samples, data, tuple(self.grid_samples), method='linear',fill_value=fill) for data in robot_dists]
                # robot_data = [gaussian_filter(griddata(samples, data, tuple(self.xy_grid), method='linear',fill_value=fill),sigma=1) for data in robot_dists]
                robot_data = [griddata(samples, data, tuple(self.xy_grid), method='linear',fill_value=fill) for data in robot_dists]
                ### matplotlib
                # with np.errstate(invalid='ignore'):state
                #     triang = mtri.Triangulation(*samples.T)
                #     robot_data = []
                #     for idx,data in enumerate(robot_dists):
                #         interp = mtri.CubicTriInterpolator(triang, data, kind='geom') # interpolate to grid
                #         zi_interp = interp(*self.xy_grid)
                #         zi_interp.data[zi_interp.mask] = np.mean(zi_interp) # mask handling
                #         zi_interp = gaussian_filter(zi_interp,sigma=1) # smooth out
                #         robot_data.append(zi_interp)
                if self.robot_figs is None:
                    self.build_robot_figs(robot_data)
                    if self.plot_samples:
                        self.build_scatter(samples)
                else:
                    levels = (np.min(robot_data),np.max(robot_data))
                    for idx,data in enumerate(robot_data):
                        self.robot_figs[idx].setImage(data, autoRange=False,levels=[0,1]) #autoLevels=True)
                        # self.robot_figs[idx].setLevels(levels)
                    if self.plot_samples:
                        self.scatter.setData(samples)
            if self.plot_future_states:
                self.pose_markers[4].setData(x=robot_plot_data[3][:,self.plot_idx[0]],y=robot_plot_data[3][:,self.plot_idx[1]])
        # except:
            # pass
        if (not throttle) and self.plot_zs: 
            if (z_mu is not None) and (z_var is not None):
                labelStyle = {'font-size': '{}pt'.format(12*self.scale)}
                z_str = ""
                if self.z_figs is None:
                    self.build_bar_graphs([z_mu,z_var])
                else:
                    scale = find_exp(np.abs(z_var).min())
                    if scale > 1:
                        scale -= 1
                        z_var *= 10**scale
                        z_str = f"x1e-{scale}"
                    [self.z_figs[idx].setOpts(height=data) for idx,data in enumerate([z_mu,z_var])]
                if self.text_scale is None:
                    self.text_scale = pg.LabelItem(z_str)
                    self.text_scale.setParentItem(self.ax1[0,2])
                    self.text_scale.anchor(itemPos=(0,0), parentPos=(0,0), offset=(40,10))
                else:
                    self.text_scale.setText(z_str)
            if self.pred_view is None:
                self.build_pred_view(img_pred)
            else:
                self.pred_view.setImage(img_pred,autoLevels=False)

        self.format_path_plot()
        self.app.processEvents()

    def project_samples(self,samples):
        if self.states is None:
            xy = samples[:,:2]
        else:
            xy = samples[:,self.plot_idx]
        return xy

    def save(self,fname,full_path=False):
        if self.robot_figs is not None:
            if not full_path:
                ext = '.svg' if self.use_svg else '.png'
                fname = self.path+fname+ext
            else:
                if self.use_svg:
                    fname = fname.replace('.png','.svg')
                else:
                    fname = fname.replace('.svg','.png')
            self.exporter.export(fname)

class TrainingPlotter(object):
    def __init__(self,render=True,path=None):
        # general
        self.render = render
        self.path = path
        for p in ['vae/', 'vae_checkpoint/']:
            if not os.path.exists(path+p):
                os.makedirs(path+p)

        self.app = pg.mkQApp()
        try:
            self.app.setFont(QtGui.QFont('Nimbus Roman')) # change font
        except:
            pass
        self.fig2 = None
        self.fig3 = None
        self.training_figs = None
        self.checkpoint_figs = None
        self.use_svg = True

    def setup_fig(self,type,data_in):
        if type == 'train':
            window_title = 'Training Figs'
            ylabel = 'VAE Training'
        else:
            window_title = 'Training Checkpoint Figs'
            ylabel = 'VAE Checkpoint'

        # set up plotting
        pg.ViewBox.suggestPadding = lambda *_: 0.0
        fig = pg.GraphicsLayoutWidget(show=self.render)
        # fig.ci.layout.setContentsMargins(0, 0, 0, 0)
        # fig.ci.layout.setSpacing(0)
        fig.setAspectLocked(True)
        fig.setWindowTitle(window_title)
        width = 550
        height = 250
        fig.setFixedWidth(width)
        fig.setFixedHeight(height)
        fig_list = []
        ax_list = []
        iter_step = data_in[-1]

        fig.addLabel(text='actual',row=0, col=1,colspan=1,size="16pt")
        fig.addLabel(text='imagined',row=0, col=2,colspan=2,size="16pt")
        fig.addLabel(text=ylabel,row=1,col=0,size="16pt",bold=True,angle=-90)

        titles = ["<html>&nbsp;</html>",f"update {iter_step[0]-iter_step[1]}", f"update {iter_step[0]}"]
        for idx, (data,title) in enumerate(zip(data_in,titles)):
            plot = fig.addPlot(row=1, col=idx+1)
            plot.hideButtons()
            plot.setTitle(title,size='16pt')
            plot.hideAxis('bottom')
            plot.hideAxis('left')
            img = pg.ImageItem(data,levels=(0.,1.))
            plot.addItem(img)
            plot.setMouseEnabled(x=False)
            plot.setMouseEnabled(y=False)
            plot.setAspectLocked(True)
            fig_list.append(img)
            ax_list.append(plot)
        qGraphicsGridLayout = fig.ci.layout
        qGraphicsGridLayout.setColumnStretchFactor(0, 0)
        qGraphicsGridLayout.setColumnStretchFactor(1, 5)
        qGraphicsGridLayout.setColumnStretchFactor(2, 5)
        qGraphicsGridLayout.setColumnStretchFactor(3, 5)

        plot_loc = QtGui.QGuiApplication.primaryScreen().availableGeometry().bottomLeft()
        if type=='train':
            # plot_loc.setY( plot_loc.y() - height*2.1)
            plot_loc.setX( plot_loc.x() + width)
            plot_loc.setY( plot_loc.y() - height)
        else:
            plot_loc.setY( plot_loc.y() - height)
        fig.move(plot_loc)

        return fig, ax_list, fig_list


    def training_update(self, args):
        # [y,y_pred_pre,y_pred_post,iter_step]=args
        iter_step = args[-1]
        if self.training_figs is None:
            self.fig2, self.ax2, self.training_figs = self.setup_fig('train',args)
            if self.use_svg:
                self.exporter2 = SVGExporter(self.fig2.scene())
            else:
                self.exporter2 = ImageExporter(self.fig2.scene())
        for fig,data in zip(self.training_figs,args):
            fig.setImage(data,autoLevels=False)
        self.ax2[1].setTitle(f'update {iter_step[0]-iter_step[1]}')
        self.ax2[2].setTitle(f'update {iter_step[0]}')
        self.app.processEvents()

    def checkpoint_update(self, args):
        # [y,y_pred_pre,y_pred_post,iter_step]=args
        iter_step = args[-1]
        if self.checkpoint_figs is None:
            self.fig3, self.ax3, self.checkpoint_figs = self.setup_fig('checkpoint',args)
            if self.use_svg:
                self.exporter3 = SVGExporter(self.fig3.scene())
            else:
                self.exporter3 = ImageExporter(self.fig3.scene())
                # self.exporter3.parameters()['width'] = 550   # (note this also affects height parameter)
        for fig,data in zip(self.checkpoint_figs,args):
            fig.setImage(data,autoLevels=False)
        self.ax3[1].setTitle(f'update {iter_step[0]-iter_step[1]}')
        self.ax3[2].setTitle(f'update {iter_step[0]}')
        self.app.processEvents()

    def save(self,fname,main_fname=None): # placeholder for old formatting
        ext = '.svg' if self.use_svg else '.png'
        if self.fig2 is not None:
            self.exporter2.export(self.path+'vae/'+fname+'_vae'+ext)
        if self.fig3 is not None:
            self.exporter3.export(self.path+'vae_checkpoint/'+fname+'_vae_checkpoint'+ext)

    def save_fig3_only(self,fname):
        self.exporter3.export(self.path+'vae_checkpoint/'+fname+'_vae_checkpoint.svg')

class DebugPlotter(object):
    def __init__(self,render=True,path=None,shared_model=False):
        # general
        self.render = render
        self.path = path
        if os.path.exists(path) == False:
            os.makedirs(path)

        self.app = pg.mkQApp()
        try:
            self.app.setFont(QtGui.QFont('Nimbus Roman')) # change font
        except:
            pass
        self.fig2 = None
        self.exporter2 = None
        self.debug_figs = None
        self.shared_model = shared_model

    def setup_fig(self,data_in,data_in_pred,seed,data_pred_seeded):
        window_title = 'Debug Figs'

        # set up plotting
        pg.ViewBox.suggestPadding = lambda *_: 0.0
        fig = pg.GraphicsLayoutWidget(show=self.render)
        # fig.ci.layout.setContentsMargins(0, 0, 0, 0)
        # fig.ci.layout.setSpacing(0)
        fig.setAspectLocked(True)
        fig.setWindowTitle(window_title)
        width = 1000
        height = 300 if self.shared_model else 200
        fig.setFixedWidth(width)
        fig.setFixedHeight(height)
        fig_list = []
        ax_list = []

        offset = 2 if self.shared_model else 1
        fig.addLabel(text=' actual ',row=1, col=0, size="16pt",bold=True,angle=-90)

        for idx, data in enumerate(data_in):
            plot = fig.addPlot(row=1, col=idx+offset)
            plot.hideButtons()
            plot.hideAxis('bottom')
            plot.hideAxis('left')
            img = pg.ImageItem(data,levels=(0.,1.))
            plot.addItem(img)
            plot.setMouseEnabled(x=False)
            plot.setMouseEnabled(y=False)
            plot.setAspectLocked(True)
            fig_list.append(img)
            ax_list.append(plot)

        fig.addLabel(text='imagined',row=2, col=0, size="16pt",bold=True,angle=-90)

        for idx, data in enumerate(data_in_pred):
            plot = fig.addPlot(row=2, col=idx+offset)
            plot.hideButtons()
            plot.setAspectLocked(True)
            plot.hideAxis('bottom')
            plot.hideAxis('left')
            img = pg.ImageItem(data,levels=(0.,1.))
            plot.addItem(img)
            plot.setMouseEnabled(x=False)
            plot.setMouseEnabled(y=False)
            fig_list.append(img)
            ax_list.append(plot)
        
        if self.shared_model:
            fig.addLabel(text=' seeded ',row=3, col=0, size="16pt",bold=True,angle=-90)
            # seed
            plot = fig.addPlot(row=3, col=1)
            plot.hideButtons()
            plot.setAspectLocked(True)
            plot.hideAxis('bottom')
            plot.hideAxis('left')
            img = pg.ImageItem(seed,levels=(0.,1.))
            plot.addItem(img)
            plot.setMouseEnabled(x=False)
            plot.setMouseEnabled(y=False)
            fig_list.append(img)
            ax_list.append(plot)

            for idx, data in enumerate(data_pred_seeded):
                plot = fig.addPlot(row=3, col=idx+offset)
                plot.hideButtons()
                plot.setAspectLocked(True)
                plot.hideAxis('bottom')
                plot.hideAxis('left')
                img = pg.ImageItem(data,levels=(0.,1.))
                plot.addItem(img)
                plot.setMouseEnabled(x=False)
                plot.setMouseEnabled(y=False)
                fig_list.append(img)
                ax_list.append(plot)


        # qGraphicsGridLayout = fig.ci.layout
        # qGraphicsGridLayout.setColumnStretchFactor(0, 0)
        # for idx in data_in_pred:
        #     qGraphicsGridLayout.setColumnStretchFactor(idx, 5)
            
        plot_loc = QtGui.QGuiApplication.primaryScreen().availableGeometry().bottomRight()
        plot_loc.setX( plot_loc.x() - width)
        plot_loc.setY( plot_loc.y() - height-50)
        fig.move(plot_loc)

        return fig, ax_list, fig_list


    def update(self,ys,y_preds,seed,y_preds_seeded):
        if self.debug_figs is None:
            self.fig2, self.ax2, self.debug_figs = self.setup_fig(ys,y_preds,seed,y_preds_seeded)
            self.exporter2 = SVGExporter(self.fig2.scene())
        else:
            for fig,data in zip(self.debug_figs,np.vstack([ys,y_preds,seed,y_preds_seeded])):
                fig.setImage(data,autoLevels=False)
        self.app.processEvents()


    def save(self,fname): 
        if (self.fig2 is not None) and (self.exporter2 is not None):
            self.exporter2.export(self.path+fname+'.svg')

