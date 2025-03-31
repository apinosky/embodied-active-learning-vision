#!/usr/bin/env python
import random
# import numpy as np
import torch

class MemoryBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def get_recent(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer[-batch_size:]
        else:
            if (self.position-batch_size) < 0:
                return self.buffer[:self.position]+self.buffer[(self.position-batch_size):]
            else:
                return self.buffer[(self.position-batch_size):self.position]

    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.position = 0
        self.buffer = []

class MemoryBuffer_torch(torch.nn.Module):
    def __init__(self, capacity, state_dim, dtype=float):
        super(MemoryBuffer_torch, self).__init__()        
        self.capacity = capacity
        self.position = 0
        self.full_buffer = False
        self.buffer = torch.empty((capacity,state_dim),dtype=dtype)

    def push(self, state):
        if (self.position + 1 ) == self.capacity:
            self.full_buffer = True
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.__len__() == 0:
            random_indices = []
        elif self.full_buffer:
            if batch_size > self.capacity:
                batch_size = self.capacity
            random_indices = torch.randperm(self.capacity)[:batch_size]
        else:
            if batch_size > self.position:
                batch_size = self.position
            random_indices = torch.randperm(self.position)[:batch_size]
        return self.buffer[random_indices, :].clone()

    def get_recent(self, batch_size):
        if self.position > batch_size:
            return self.buffer[self.position-batch_size:self.position].clone()
        else:
            if self.full_buffer:
                return torch.vstack([self.buffer[:self.position],
                                  self.buffer[self.position-batch_size:]])
            else:
                return self.buffer[:self.position].clone()

    def get_all(self):
        if self.full_buffer:
            return self.buffer.clone()
        else:
            return self.buffer[:self.position].clone()

    def __len__(self):
        if self.full_buffer:
            return self.capacity
        else:
            return self.position

    def seed(self,seed):
        torch.manual_seed(seed)

    def reset(self):
        self.position = 0
        self.full_buffer = False


class AvoidDist(torch.nn.Module):
    def __init__(self,state_dim,capacity=1000,invert=True):
        super(AvoidDist, self).__init__()        
        self.state_dim = state_dim
        self.capacity = capacity
        self.invert = invert

        self.reset()


    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        val = torch.ones(x.shape[0])
        val /= torch.sum(val)
        val += 1e-5
        return val

    def pdf(self,samples):
        if self.position > 0 or self.full_buffer:
            args = {}
            if self.full_buffer:
                means = self.env_path.clone()
                stds = self.env_path_val.clone()
            else:
                means = self.env_path[:self.position].clone()
                stds = self.env_path_val[:self.position].clone()

            dist = torch.mean(torch.exp(-0.5 * torch.sum(torch.square(means.unsqueeze(0)-samples.unsqueeze(1))/stds.unsqueeze(0), 2)),1)
            if self.invert:
                dist = -dist+torch.max(dist)+torch.min(dist) # invert distribution
            return dist
        else:
            return self.init_uniform_grid(samples)

    def push(self,state,val):
        if (self.position + 1 ) == self.capacity:
            self.full_buffer = True
            print('buffer full')
        self.env_path[self.position] = state
        self.env_path_val[self.position] = val
        self.position = (self.position + 1) % self.capacity

    def reset(self):
        self.full_buffer = False
        self.position = 0
        self.env_path = torch.empty([self.capacity,self.state_dim])
        self.env_path_val = torch.empty([self.capacity,self.state_dim])

    def get_buffer(self):
        if self.full_buffer:
            return self.env_path.clone(),self.env_path_val.clone()
        else:
            return self.env_path[:self.position].clone(),self.env_path_val[:self.position].clone()
