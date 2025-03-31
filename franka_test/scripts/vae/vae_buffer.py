#!/usr/bin/env python

import random
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, use_torch=False, learn_force=False):
        self.capacity = capacity
        self.use_torch = use_torch
        self.learn_force = learn_force
        self.paused = False
        self.reset()

    def reset(self):
        self.buffer = []
        self.position = 0
        self.total_steps = 0

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def push(self, x, y, force=None):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        if self.learn_force:
            self.buffer[self.position] = (x, y, force)
        else:
            self.buffer[self.position] = (x, y)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        if self.learn_force:
            if self.use_torch:
                x, y, force  = map(torch.stack, zip(*batch))
            else:
                x, y, force  = map(np.stack, zip(*batch))
            return x, y, force
        else:
            if self.use_torch:
                x, y  = map(torch.stack, zip(*batch))
            else:
                x, y  = map(np.stack, zip(*batch))
            return x, y

    def get_last(self,num=1):
        batch = self.buffer[self.position-num:self.position]
        if self.learn_force:
            if self.use_torch:
                x, y, force  = map(torch.stack, zip(*batch))
            else:
                x, y, force  = map(np.stack, zip(*batch))
            return x, y, force
        else:
            if self.use_torch:
                x, y  = map(torch.stack, zip(*batch))
            else:
                x, y  = map(np.stack, zip(*batch))
            return x, y

    def __len__(self):
        return len(self.buffer)

    @property
    def explr_step(self):
        return self.total_steps
    def load(self,data):
        if isinstance(data,list):
            self.buffer = data
        elif isinstance(data,object):
            data = data[np.logical_not(data==None)].tolist()
            self.buffer = data
        else:
            raise ValueError('valid data types are list or numpy array (object)')
        self.position = len(self.buffer)

    def __repr__(self):
        msg = f'capacity: {self.capacity}\n'
        msg += f'position: {self.position}\n'
        msg += f'buffer len: {self.__len__()}\n'
        return msg

class zBufferTorch(torch.nn.Module):
    __attributes__ = ['capacity','device','dtype']
    def __init__(self, capacity: int, z_dim: int, dtype: int, device: str):
        super(zBufferTorch,self).__init__()
        self.device = device
        self.dtype = dtype
        self.capacity = capacity

        self.z_buffer = torch.empty(capacity,z_dim,dtype=dtype,device=device)
        self.init = torch.tensor([False])
        self.full_buffer = torch.tensor([False])
        self.position = torch.tensor([0])

        self.reset()

    def reset(self):
        self.init[0] = False
        self.full_buffer[0] = False
        self.position[0] = 0

    def push_batch(self,z):
        self.init[0] = True
        num_items=z.shape[0]
        if (not self.full_buffer) and ((self.position + num_items ) >= self.capacity):
            self.full_buffer[0] = True
        self.z_buffer[self.position:self.position+num_items] = z
        self.position[0] = (self.position + num_items) % self.capacity

    def push(self,z):
        self.init[0] = True
        if (not self.full_buffer) and ((self.position + 1 ) >= self.capacity):
            self.full_buffer[0] = True
        self.z_buffer[self.position] = z
        self.position[0] = (self.position + 1) % self.capacity

    def isfull(self):
        return self.full_buffer

    def __len__(self):
        if self.full_buffer:
            return self.capacity
        else:
            return self.position

    def get_samples(self):
        if self.init:
            if self.full_buffer:
                return self.z_buffer.detach().clone()
            else:
                return self.z_buffer[:self.position].detach().clone()

    def extra_repr(self):
        return  '\n'.join(('{} = {}'.format(item, self.__dict__[item].shape if isinstance(self.__dict__[item],torch.Tensor) else self.__dict__[item]) for item in self.__attributes__))

class ReplayBufferTorch(torch.nn.Module):
    __attributes__ = ['x_buffer','y_buffer','device','dtype']
    def __init__(self, capacity: int, x_dim: int, y_dim: int, dtype: int, device: str, learn_force: bool, beta_capacity: int = 25, buffer_device: str = 'cpu',world_size: int = 1, batch_size: int = 10):
        super(ReplayBufferTorch,self).__init__()
        self.device = device
        self.buffer_device = buffer_device
        self.dtype = dtype
        torch.set_default_dtype(dtype)
        self.learn_force = learn_force

        self.capacity = capacity
        self.position = torch.tensor([0])
        self.total_steps = torch.tensor([0])

        # how to pull batches for multiple processes
        self._world_size = torch.tensor(world_size,dtype=torch.int)
        self._batch_size = torch.tensor(batch_size,dtype=torch.int)
        batch_per_proc = int(batch_size/world_size)
        self._batch_indices_capacity = 10
        self._batch_indices = torch.zeros(self._batch_indices_capacity,world_size,batch_per_proc,dtype=torch.int,device=self.buffer_device)
        self._batch_indices_position = torch.zeros(world_size+1,dtype=torch.int,device=self.buffer_device)
        self._batch_indices_weighted = torch.zeros(self._batch_indices_capacity,world_size,batch_per_proc,dtype=torch.int,device=self.buffer_device)
        self._batch_indices_weighted_position = torch.zeros(world_size+1,dtype=torch.int,device=self.buffer_device)

        self.full_buffer = torch.tensor([False])
        self.x_buffer = torch.empty(capacity,x_dim,dtype=dtype,device=self.buffer_device)
        self.y_buffer = torch.empty(capacity,*y_dim,dtype=dtype,device=self.buffer_device)
        self.y_var_buffer = torch.empty(capacity,dtype=dtype,device=self.buffer_device)
        self.paused = torch.tensor([False])

        self.beta_capacity = beta_capacity
        # self.last_beta = 0.
        # self.max_beta_ramp = 0.001
        self.full_beta = torch.tensor([False])
        self.beta_position = torch.tensor([0])
        self.beta = torch.zeros(self.beta_capacity,dtype=dtype,device=self.buffer_device)
        self.gamma = torch.zeros(self.beta_capacity,dtype=dtype,device=self.buffer_device)
        self.explr_ind = torch.tensor([0])

        if self.learn_force:
            self.force_buffer = torch.empty(capacity,1,dtype=dtype,device=self.buffer_device)
            self.__attributes__.append('force_buffer')

        self.reset()

    def update_hyperparams(self,explr_ind,val1,val2=0.): 
        # error checking 
        error = False
        for val in [val1,val2]:
            if isinstance(val,torch.Tensor): 
                error = error or torch.isnan(val) or torch.isinf(val)
            else: 
                error = error or np.isnan(val) or np.isinf(val) 
        if error:
            print('invalid hyperparams',val)
        else:
            self.explr_ind[0]=explr_ind
            for val,beta in zip([val1,val2],[self.beta,self.gamma]):
                if isinstance(val1,torch.Tensor): 
                    beta[self.beta_position] = val.detach().clone().to(dtype=self.dtype,device=self.buffer_device)
                elif isinstance(val1,(int,float)):
                    beta[self.beta_position] = val
                else: 
                    beta[self.beta_position] = val[0]
            if (self.beta_position + 1 ) == self.beta_capacity:
                self.full_beta[0] = True
            self.beta_position += 1
            self.beta_position.remainder_(self.beta_capacity)

    def get_xi(self): 
        if self.full_buffer:
            y_vars =  self.y_var_buffer[:self.capacity]
        else:
            y_vars =  self.y_var_buffer[:self.position]
        y_vars = torch.clamp(y_vars,min=np.exp(-10))
        # print(y_vars.min(),y_vars.mean(),y_vars.max())
        return y_vars.mean()/y_vars.max()*10 

    def get_hyperparams(self): 
        if self.full_beta:
            tmp_beta = self.beta.mean().item()
            tmp_gamma = self.gamma.mean().item()
        else:
            tmp_beta = self.beta[:self.beta_position].mean().item()
            tmp_gamma = self.gamma[:self.beta_position].mean().item()
        # if abs(tmp_beta - self.last_beta) > self.max_beta_ramp: 
        #     tmp_beta = self.last_beta + np.clip(tmp_beta-self.last_beta,-self.max_beta_ramp,self.max_beta_ramp) 
        # self.last_beta = tmp_beta
        return self.explr_ind.item(), tmp_beta,tmp_gamma

    def reset(self):
        self.full_buffer[0] = False
        self.position[0] = 0
        self.total_steps[0] = 0

    def pause(self):
        self.paused[0] = True

    def resume(self):
        self.paused[0] = False

    def is_shared(self):
        check_force = True
        if self.learn_force:
            check_force = self.force_buffer.is_shared()
        return all([self.paused.is_shared(),
                   self.position.is_shared(),
                   self.total_steps.is_shared(),
                   self.full_buffer.is_shared(),
                   self.x_buffer.is_shared(),
                   self.y_var_buffer.is_shared(),
                   self.y_buffer.is_shared(),
                   self.beta_position.is_shared(),
                   self.beta.is_shared(),
                   self.gamma.is_shared(),
                   self.explr_ind.is_shared(),
                   check_force,
                   self._world_size.is_shared(),
                   self._batch_size.is_shared(),
                   self._batch_indices.is_shared(),
                   self._batch_indices_position.is_shared(),
                   self._batch_indices_weighted.is_shared(),
                   self._batch_indices_weighted_position.is_shared(),
                   ])

    def share_memory(self):
        self.paused.share_memory_()
        self.position.share_memory_()
        self.total_steps.share_memory_()
        self.full_buffer.share_memory_()
        self.x_buffer.share_memory_()
        self.y_var_buffer.share_memory_()
        self.y_buffer.share_memory_()
        self.beta_position.share_memory_()
        self.beta.share_memory_()
        self.gamma.share_memory_()
        self.explr_ind.share_memory_()

        self._world_size.share_memory_()
        self._batch_size.share_memory_()
        self._batch_indices.share_memory_()
        self._batch_indices_position.share_memory_()
        self._batch_indices_weighted.share_memory_()
        self._batch_indices_weighted_position.share_memory_()

        if self.learn_force:
            self.force_buffer.share_memory_()

    def push(self, x, y, force=None):
        if (self.position + 1 ) == self.capacity:
            self.full_buffer[0] = True
        self.x_buffer[self.position] = x.to(device=self.buffer_device)
        self.y_buffer[self.position] = y.to(device=self.buffer_device)
        self.y_var_buffer[self.position] = y.flatten().var().to(device=self.buffer_device)
        if self.learn_force:
            self.force_buffer[self.position] = force.to(device=self.buffer_device)
        self.position += 1
        self.position.remainder_(self.capacity)
        self.total_steps += 1

    def push_batch(self, x, y,force=None):
        num_items=x.shape[0]
        assert num_items == y.shape[0]
        if (self.position + num_items ) >= self.capacity:
            self.full_buffer[0] = True
        self.x_buffer[self.position:self.position+num_items] = x.to(device=self.buffer_device)
        self.y_buffer[self.position:self.position+num_items] = y.to(device=self.buffer_device)
        self.y_var_buffer[self.position:self.position+num_items] = y.reshape(y.shape[0],-1).var(1).to(device=self.buffer_device)
        if self.learn_force:
            self.force_buffer[self.position:self.position+num_items] = force.to(device=self.buffer_device)
        self.position += num_items
        self.position.remainder_(self.capacity)
        self.total_steps += num_items

    def sample(self, batch_size, weighted=False):
        num_samps = self.capacity if self.full_buffer else self.position.item()
        if weighted:
            weights = torch.clamp(torch.arange(num_samps,dtype=self.dtype),min=num_samps/2)
        else:
            weights = torch.ones(num_samps)
        weights /= weights.sum()
        random_indices = torch.multinomial(weights,batch_size)
        # if self.full_buffer:
        #     random_indices = torch.randperm(self.capacity)[:batch_size]
        # else:
        #     random_indices = torch.randperm(self.position.item())[:batch_size]
        out = [self.x_buffer[random_indices, :].detach().clone().to(device=self.device),
               self.y_buffer[random_indices, :].detach().clone().to(device=self.device)]
        if self.learn_force:
            out.append(self.force_buffer[random_indices, :].detach().clone().to(device=self.device))
        return out + [random_indices]

    def check_batch(self, rank=0):
        if rank == 0:  # pull new batch
            num_samps = self.capacity if self.full_buffer else self.position.item()
            ## weighted
            weights = torch.clamp(torch.arange(num_samps,dtype=self.dtype),min=num_samps/2)
            weights /= weights.sum()
            new_random_indices = torch.multinomial(weights,self._batch_size)
            new_random_indices = new_random_indices.reshape(self._world_size,-1)
            new_pos = self._batch_indices_weighted_position[-1]
            self._batch_indices_weighted[new_pos] = new_random_indices
            new_pos += 1
            self._batch_indices_weighted_position.remainder_(self._batch_indices_capacity)
            ## unweighted
            weights = torch.ones(num_samps)
            weights /= weights.sum()
            for _ in range(2):
                new_random_indices = torch.multinomial(weights,self._batch_size)
                new_random_indices = new_random_indices.reshape(self._world_size,-1)
                new_pos = self._batch_indices_position[-1]
                self._batch_indices[new_pos] = new_random_indices
                new_pos += 1
                self._batch_indices_position.remainder_(self._batch_indices_capacity)
            return True
        else:
            w_pos = self._batch_indices_weighted_position[rank]
            w_random_indices = self._batch_indices_weighted[w_pos,rank]

            pos = self._batch_indices_position[rank]
            random_indices = self._batch_indices[pos,rank]
            return not ( (w_random_indices == 0).all() and (random_indices == 0).all())

    def sample_batch(self, rank=0, weighted=False):
        if weighted:
            pos = self._batch_indices_weighted_position[rank]
            random_indices = self._batch_indices_weighted[pos,rank]
        else:
            pos = self._batch_indices_position[rank]
            random_indices = self._batch_indices[pos,rank]
        if (random_indices == 0).all(): # new batch not ready
            if rank == 0: 
                self.check_batch()
            else:
                # print('out of sync, pulling batch',rank)
                return self.sample(len(random_indices),weighted) # something got out of sync, so just pull some data
        out = [self.x_buffer[random_indices, :].detach().clone().to(device=self.device),
            self.y_buffer[random_indices, :].detach().clone().to(device=self.device)]
        if self.learn_force:
            out.append(self.force_buffer[random_indices, :].detach().clone().to(device=self.device))
        # clear indices
        pos += 1
        random_indices[:] = 0
        if weighted:
            self._batch_indices_weighted_position.remainder_(self._batch_indices_capacity)
        else:
            self._batch_indices_position.remainder_(self._batch_indices_capacity)
        return out

    def __len__(self):
        if self.full_buffer:
            return self.capacity
        else:
            return self.position

    @property
    def explr_step(self):
        return self.total_steps.item()
    
    def get_last(self):
        out = [self.x_buffer[self.position-1, :].detach().clone().to(device=self.device),
               self.y_buffer[self.position-1, :].detach().clone().to(device=self.device)]
        if self.learn_force:
            out.append(self.force_buffer[self.position-1, :].detach().clone().to(device=self.device))
        return out
    
    def get_all_x(self): 
        if self.full_buffer:
            return self.x_buffer.detach().clone().to(device=self.device)
        else: 
            return self.x_buffer[:self.position].detach().clone().to(device=self.device)

    def extra_repr(self):
        return  '\n'.join(('{} = {}'.format(item, self.__dict__[item].shape if isinstance(self.__dict__[item],torch.Tensor) else self.__dict__[item]) for item in self.__attributes__))

