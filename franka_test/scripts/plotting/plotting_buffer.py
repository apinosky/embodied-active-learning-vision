#!/usr/bin/env python

import random
import numpy as np
import torch
import time

class PlottingBufferTorch(torch.nn.Module):
    __attributes__ = ['state_buffer','sensor_data_buffer','pq_samples_buffer','p_buffer','q_buffer','z_mu_buffer','z_var_buffer','sensor_data_pred_buffer','iter_buffer']
    def __init__(self,capacity,x_dim,y_dim,z_dim,num_target_samples,explr_dim,horizon,use_z=True,device='cpu',dtype=torch.float32):
        super(PlottingBufferTorch,self).__init__()
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
        self.device = device
        self.capacity = capacity
        self.position = torch.tensor([0])
        self.next_position = torch.tensor([0])
        self.full_buffer = torch.tensor([False])
        self.state_buffer = torch.empty(capacity,x_dim)
        self.force_buffer = torch.empty(capacity,1)
        self.sensor_data_buffer = torch.empty(capacity,*y_dim)
        self.sensor_data_pred_buffer = torch.empty(capacity,*y_dim)
        self.pq_samples_buffer = torch.empty(capacity,num_target_samples,explr_dim)
        self.p_buffer = torch.empty(capacity,num_target_samples)
        self.q_buffer = torch.empty(capacity,num_target_samples)
        self.p_buffer_smooth = torch.empty(capacity,num_target_samples)
        self.q_buffer_smooth = torch.empty(capacity,num_target_samples)
        self.cost_buffer = torch.empty(capacity)
        self.future_state_buffer = torch.empty(capacity,horizon+1,x_dim)
        if use_z:
            self.z_mu_buffer = torch.empty(capacity,z_dim)
            self.z_var_buffer = torch.empty(capacity,z_dim)
        self.iter_buffer = torch.empty(capacity,2,dtype=torch.int64)
        self.dtype = dtype
        self.use_z = use_z

    def share_memory(self):
        self.position.share_memory_()
        self.full_buffer.share_memory_()
        self.state_buffer.share_memory_()
        self.force_buffer.share_memory_()
        self.sensor_data_buffer.share_memory_()
        self.sensor_data_pred_buffer.share_memory_()
        self.pq_samples_buffer.share_memory_()
        self.p_buffer.share_memory_()
        self.q_buffer.share_memory_()
        self.p_buffer_smooth.share_memory_()
        self.q_buffer_smooth.share_memory_()
        self.cost_buffer.share_memory_()
        self.future_state_buffer.share_memory_()
        if self.use_z:
            self.z_mu_buffer.share_memory_()
            self.z_var_buffer.share_memory_()
        self.iter_buffer.share_memory_()

    def push(self, sensor_data, state, force, robot_data, z_mu, z_var, sensor_data_pred, count):
        if not(isinstance(sensor_data,torch.Tensor)):
            sensor_data = torch.as_tensor(sensor_data)
        if not(isinstance(sensor_data_pred,torch.Tensor)):
            sensor_data_pred = torch.as_tensor(sensor_data_pred)
        if not(isinstance(state,torch.Tensor)):
            state = torch.as_tensor(state)
        if not(isinstance(force,torch.Tensor)):
            force = torch.as_tensor(force)
        if (self.position + 1 ) == self.capacity:
            self.full_buffer[0] = True
        self.sensor_data_buffer[self.position] = sensor_data.to(self.device)
        self.state_buffer[self.position] = state.to(self.device)
        self.force_buffer[self.position] = force.to(self.device)
        self.pq_samples_buffer[self.position] = torch.as_tensor(robot_data[0])
        self.p_buffer[self.position] = torch.as_tensor(robot_data[1])
        self.q_buffer[self.position] = torch.as_tensor(robot_data[2])
        self.p_buffer_smooth[self.position] = torch.as_tensor(robot_data[4])
        self.q_buffer_smooth[self.position] = torch.as_tensor(robot_data[5])
        self.cost_buffer[self.position] = torch.as_tensor(robot_data[6])
        self.future_state_buffer[self.position] = torch.as_tensor(robot_data[3])
        if self.use_z:
            if not(isinstance(z_mu,torch.Tensor)):
                z_mu = torch.as_tensor(state)
                if not(isinstance(z_var,torch.Tensor)):
                    z_var = torch.as_tensor(state)
            self.z_mu_buffer[self.position] = z_mu.to(self.device)
            self.z_var_buffer[self.position] = z_var.to(self.device)
        self.sensor_data_pred_buffer[self.position] = sensor_data_pred.to(self.device)
        self.iter_buffer[self.position] = torch.as_tensor(count)
        self.position += 1
        self.position.remainder_(self.capacity)

    def __len__(self):
        return np.abs(self.position - self.next_position)

    def get_next(self):
        if self.use_z:
            out =  [self.sensor_data_buffer[self.next_position].squeeze(),
                    self.state_buffer[self.next_position].squeeze(),
                    self.force_buffer[self.next_position].squeeze(),
                    [self.pq_samples_buffer[self.next_position].squeeze().detach().numpy(),
                    self.p_buffer[self.next_position].squeeze().detach().numpy(),
                    self.q_buffer[self.next_position].squeeze().detach().numpy(),
                    self.future_state_buffer[self.next_position].squeeze().detach().numpy(),
                    self.p_buffer_smooth[self.next_position].squeeze().detach().numpy(),
                    self.q_buffer_smooth[self.next_position].squeeze().detach().numpy(),
                    self.cost_buffer[self.next_position].squeeze().detach().numpy()],
                    self.z_mu_buffer[self.next_position].squeeze(),
                    self.z_var_buffer[self.next_position].squeeze(),
                    self.sensor_data_pred_buffer[self.next_position].squeeze(),
                    self.iter_buffer[self.next_position].squeeze().detach().numpy()]
        else:
            out =  [self.sensor_data_buffer[self.next_position].squeeze(),
                    self.state_buffer[self.next_position].squeeze(),
                    self.force_buffer[self.next_position].squeeze(),
                    [self.pq_samples_buffer[self.next_position].squeeze().detach().numpy(),
                    self.p_buffer[self.next_position].squeeze().detach().numpy(),
                    self.q_buffer[self.next_position].squeeze().detach().numpy(),
                    self.future_state_buffer[self.next_position].squeeze().detach().numpy(),
                    self.p_buffer_smooth[self.next_position].squeeze().detach().numpy(),
                    self.q_buffer_smooth[self.next_position].squeeze().detach().numpy(),
                    self.cost_buffer[self.next_position].squeeze().detach().numpy()],
                    None,
                    None,
                    None,
                    self.iter_buffer[self.next_position].squeeze().detach().numpy()]

        self.next_position += 1
        self.next_position.remainder_(self.capacity)
        return out

    def extra_repr(self):
        return  '\n'.join(('{} = {}'.format(item, self.__dict__[item].shape if isinstance(self.__dict__[item],torch.Tensor) else self.__dict__[item]) for item in self.__attributes__))

