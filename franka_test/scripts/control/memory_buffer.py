#!/usr/bin/env python
import random
import numpy as np

class MemoryBuffer:
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

class MemoryBuffer_numpy:
    def __init__(self, capacity, state_dim, dtype=float):
        self.capacity = capacity
        self.position = 0
        self.full_buffer = False
        self.buffer = np.empty((capacity,state_dim),dtype=dtype)
        self.npr = np.random.RandomState()

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
            random_indices = self.npr.choice(self.capacity,size=batch_size,replace=False)
        else:
            if batch_size > self.position:
                batch_size = self.position
            random_indices = self.npr.choice(self.position,size=batch_size,replace=False) # (batch_size > self.position))
        return self.buffer[random_indices, :]

    def get_recent(self, batch_size):
        if self.position > batch_size:
            return self.buffer[self.position-batch_size:self.position].copy()
        else:
            if self.full_buffer:
                return np.vstack([self.buffer[:self.position],
                                  self.buffer[self.position-batch_size:]])
            else:
                return self.buffer[:self.position].copy()

    def get_all(self):
        if self.full_buffer:
            return self.buffer.copy()
        else:
            return self.buffer[:self.position].copy()

    def __len__(self):
        if self.full_buffer:
            return self.capacity
        else:
            return self.position

    def seed(self,seed):
        # self.npr = np.random.default_rng(seed)
        self.npr = np.random.RandomState(seed)

    def reset(self):
        self.position = 0
        self.full_buffer = False

from scipy.stats import multivariate_normal as mvn

class AvoidDist(object):
    def __init__(self,state_dim,explr_idx=[0,1],capacity=1000,invert=True):
        self.state_dim = state_dim
        self.explr_idx = explr_idx
        self.capacity = capacity
        self.invert = invert

        self.reset()


    def init_uniform_grid(self, x):
        assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        val = np.ones(x.shape[0])
        val /= np.sum(val)
        val += 1e-5
        return val

    def pdf(self,samples):
        if self.position > 0 or self.full_buffer:
            args = {}
            if self.full_buffer:
                means = self.env_path.copy()[:,self.explr_idx]
                stds = self.env_path_val.copy()[:,self.explr_idx]
            else:
                means = self.env_path[:self.position].copy()[:,self.explr_idx]
                stds = self.env_path_val[:self.position].copy()[:,self.explr_idx]
            dist = np.mean([mvn.pdf(samples, mean=mu, cov=std) for mu,std in zip(means,stds)],axis=0)
            if self.invert:
                dist = -dist+np.max(dist)+np.min(dist) # invert distribution
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
        self.env_path = np.empty([self.capacity,self.state_dim])
        self.env_path_val = np.empty([self.capacity,self.state_dim])

    def get_buffer(self):
        if self.full_buffer:
            return self.env_path.copy(),self.env_path_val.copy()
        else:
            return self.env_path[:self.position].copy(),self.env_path_val[:self.position].copy()
