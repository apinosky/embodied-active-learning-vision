#!/usr/bin/env python

import numpy as np
import torch
from torch import nn
from termcolor import cprint
import time
from .vae_buffer import zBufferTorch
from .vae_utils import *

class VAE(nn.Module):
    # Image VAE code/setup
    def __init__(self, img_dim, z_dim, s_dim, hidden_dim=[256,128],
                    y_logvar_dim=1,force_dim=1,
                    CNNdict={'kernel_size':[5,5],'stride':[3,2],'channels':[10,10]},dx=False):
        # input/output dim=camimg_dim, z_dim=# of latent space, s_dim=dim of conditional [x,y]
        super(VAE, self).__init__()

        input_dim, inner_shape = get_input_dim(img_dim,CNNdict)
        padding = get_padding(img_dim,CNNdict)
        self.ylogvar_dim = np.prod(y_logvar_dim)
        encoder_input_dim  = int(input_dim + s_dim + force_dim)
        encoder_output_dim = int(z_dim * 2)
        decoder_input_dim  = int(z_dim+s_dim)
        decoder_output_dim = int(input_dim + self.ylogvar_dim + force_dim) #*2)

        if (y_logvar_dim == 1) or (y_logvar_dim == img_dim[0]):
            self.reshape_var = reshape1D(img_dim,y_logvar_dim)
        elif len(y_logvar_dim) == 2:
            self.reshape_var = reshape2D(img_dim,y_logvar_dim)
        else:
            raise ValueError('not sure what to do with specified input dim')

        if CNNdict is not None: 
            img_encoder_layers = []
            in_channel_layers = []
            in_channels = int(img_dim[0])
            for kernel_size,out_chanels,stride in zip(CNNdict['kernel_size'],CNNdict['channels'],CNNdict['stride']):
                in_channel_layers.append(in_channels)
                img_encoder_layers.append(
                            nn.Conv2d(in_channels=in_channels,
                                    out_channels=out_chanels,
                                    kernel_size=kernel_size, # square kernel
                                    stride=stride),
                )
                img_encoder_layers.append(nn.ReLU())
                in_channels = out_chanels
            img_encoder_layers[-1] = Flatten()
            self.img_encoder = nn.Sequential(*img_encoder_layers)

            img_decoder_layers = []
            img_decoder_layers.append(UnFlatten(shape=inner_shape))
            for kernel_size,in_channels,stride,out_channels,pad in zip(reversed(CNNdict['kernel_size']),reversed(CNNdict['channels']),
                                                                        reversed(CNNdict['stride']),reversed(in_channel_layers),padding):
                img_decoder_layers.append(
                    nn.ConvTranspose2d(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=kernel_size, # square kernel
                                    stride=stride,
                                    output_padding=int(pad) # fix output size with "even sized" images (e.g. 38x38)
                                    ),
                )
                img_decoder_layers.append(nn.ReLU())
            self.img_decoder = nn.Sequential(*img_decoder_layers[:-1])
        else: 
            self.img_encoder = Flatten()
            self.img_decoder = UnFlatten(shape=inner_shape)

        encoder_layers = []
        in_dim = encoder_input_dim
        for out_dim in hidden_dim:
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = out_dim
        encoder_layers.append(nn.Linear(in_dim, encoder_output_dim))
        self.encode  = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = decoder_input_dim
        for out_dim in reversed(hidden_dim):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, decoder_output_dim))
        self.decode  = nn.Sequential(*decoder_layers)

        self.z_dim = z_dim
        self.img_dim = img_dim
        self.force_dim = force_dim
        self.input_dim = torch.tensor(input_dim)
        self.init=torch.tensor([False])
        self.learn_force = True
        self.device='cpu'
        self.dtype=torch.float32
        seed_x = torch.zeros(1,s_dim,dtype=self.dtype,device=self.device)
        seed_y = torch.zeros(1,*img_dim,dtype=self.dtype,device=self.device)
        seed_force = torch.zeros(1,force_dim,dtype=self.dtype,device=self.device)
        z_samples = torch.zeros(1,z_dim,dtype=self.dtype,device=self.device)
        self.register_buffer('seed_x',seed_x)
        self.register_buffer('seed_y',seed_y)
        self.register_buffer('seed_force',seed_force)
        self.register_buffer('z_samples',z_samples)
        self.register_buffer('learning_ind',torch.tensor([0]))
        self.use_buffer = False
        self.use_chunk_decode = False
        self.logvar_lims = (-10,2) # (-10,2) # (-5,2)
        self.dx = dx

        self.reset()

    def reset(self):
        # self.encode.apply(weights_init)
        # self.decode.apply(weights_init)
        # self.img_encoder.apply(weights_init)
        # self.img_decoder.apply(weights_init)
        if self.use_chunk_decode:
            self.build_chunk_decoder()
        if self.use_buffer:
            self.build_z_buffer()

    # def increase_z_dim(self,size=1):
    #     enc_z = self.encode[-1]
    #     enc_z.out_features += size*2
    #     enc_z.weight = torch.nn.Parameter(torch.cat([enc_z.weight,torch.randn(size*2,enc_z.in_features)],axis=0))
    #     enc_z.bias = torch.nn.Parameter(torch.cat([enc_z.bias,torch.randn(size*2)]))

    #     dec_z = self.decode[0]
    #     dec_z.in_features += size
    #     dec_z.weight = torch.nn.Parameter(torch.cat([dec_z.weight,torch.randn(dec_z.out_features,size)],axis=1))
    #     dec_z.bias = torch.nn.Parameter(torch.cat([dec_z.bias,torch.randn(size)]))

    #     self.z_dim += 1
    #     self.reset()


    @torch.no_grad()
    def build_z_buffer(self,z_mem=5):
        z_batch = 1,  #50
        self.z_buff = zBufferTorch(z_batch*z_mem,self.z_dim,self.dtype,self.device)
        self.use_buffer = True

    def build_chunk_decoder(self,num_threads=20):
        self.num_chunks = num_threads
        self.chunk_decode = torch.jit.script(chunk_decode(self.decode,num_threads).eval())
        self.chunk_decode = torch.jit.freeze(self.chunk_decode)
        self.use_chunk_decode = True

    def reparameterize(self, mu, logvar):
        if self.training:
            var = torch.exp(logvar*0.5)
            eps = torch.randn_like(var)
            z = mu + eps*var
            return z
        else:
            return mu

    def split_y_out(self,y_out):
        # y_pred, y_logvar, force_pred, force_logvar
        ## combo var
        splits = [self.ylogvar_dim,self.ylogvar_dim+self.force_dim]
        return y_out[:,splits[1]:], y_out[:,:splits[0]], y_out[:,splits[0]:splits[1]], y_out[:,:splits[0]]
        ## separate force var 
        # splits = [self.ylogvar_dim,self.ylogvar_dim+self.force_dim,self.ylogvar_dim+self.force_dim*2]
        # return y_out[:,splits[2]:], y_out[:,:splits[0]], y_out[:,splits[0]:splits[1]], y_out[:,splits[1]:splits[2]]


    def forward(self, x, y, force, x_decode=torch.empty(0)):
        y_transform = self.img_encoder(y)
        enc_input = torch.cat([y_transform, force, x], dim=1) # y_transform
        z_out = self.encode(enc_input)
        z_mu, z_logvar = z_out[:,:self.z_dim], z_out[:, self.z_dim:]
        z_logvar = torch.clamp(z_logvar,*self.logvar_lims)
        z_samples = self.reparameterize(z_mu, z_logvar)

        if self.dx : 
            y_out = self.decode(torch.cat([z_samples, torch.zeros_like(x)], dim=1))
        else: 
            y_out = self.decode(torch.cat([z_samples, x], dim=1))
        y_pred, y_logvar, force_pred, force_logvar = self.split_y_out(y_out)

        img_pred = self.img_decoder(y_pred)
        img_logvar = torch.clamp(y_logvar, *self.logvar_lims)
        img_logvar = self.reshape_var(img_logvar)
        force_logvar = torch.clamp(force_logvar, *self.logvar_lims)

        if not(x_decode.shape==torch.Size([0])):
            y_out2 = self.decode(torch.cat([z_samples, x_decode], dim=1))
            y_pred2, y_logvar2, force_pred_decode, force_logvar2 = self.split_y_out(y_out2)

            img_pred_decode = self.img_decoder(y_pred2)
            img_logvar_decode = torch.clamp(y_logvar2, *self.logvar_lims)
            img_logvar_decode = self.reshape_var(img_logvar_decode)
            force_logvar_decode = torch.clamp(force_logvar2, *self.logvar_lims)

            return img_pred, img_logvar, z_mu, z_logvar, z_samples, force_pred, force_logvar, img_pred_decode, img_logvar_decode, force_pred_decode, force_logvar_decode
        else: 
            return img_pred, img_logvar, z_mu, z_logvar, z_samples, force_pred, force_logvar, x_decode, x_decode, x_decode, x_decode

    def decode_samples_only(self, samples, get_pred=False, get_force=False):
        x_decode = (samples).to(device=self.device,dtype=self.dtype)
        if self.dx: 
            x_decode = x_decode - self.seed_x
        if self.use_buffer:
            z_buff = self.z_buff.get_samples()
            z_mem = z_buff.shape[0]
            num_samps = samples.shape[0]
            latent_space = torch.vstack([torch.cat([zs.repeat(x_decode.shape[0], 1), x_decode], dim=1) for zs in z_buff])
        else:
            z_samples = self.z_samples.repeat(x_decode.shape[0], 1)
            latent_space = torch.cat([z_samples, x_decode], dim=1)

        if self.use_chunk_decode:
            y_out = self.chunk_decode(latent_space)
        else:
            y_out = self.decode(latent_space)

        y_pred, y_logvar, force_pred, force_logvar = self.split_y_out(y_out)
        var_data = torch.clamp(y_logvar, *self.logvar_lims)
        force_logvar = torch.clamp(force_logvar, *self.logvar_lims)
        if self.use_buffer:
            var_data = var_data.reshape(z_mem,num_samps,self.ylogvar_dim)
            var_data = torch.mean(var_data,0)
        img_logvar = self.reshape_var(var_data)
        if get_pred and get_force:
            img_pred = self.img_decoder(y_pred)
            return img_pred, img_logvar, force_pred, force_logvar
        elif get_pred and not(get_force): 
            img_pred = self.img_decoder(y_pred)
            return img_pred, img_logvar
        elif get_force and not(get_pred): 
            return force_logvar
        else: 
            return img_logvar

    # Functions for Target Dist
    def init_uniform_grid(self, x):
        # assert len(x.shape) > 1, 'Input needs to be a of size N x n'
        # val = torch.ones(x.shape[0])
        # val = val / torch.sum(val)
        # val += 1e-5
        val = x.sum(1)**0
        return val

    @torch.no_grad()
    def update_dist(self, xr, y, force):
        out = self.forward(xr.to(device=self.device), y.to(device=self.device), force.to(device=self.device))
        self.seed_x[0] = xr.detach().clone()
        self.seed_y[0] = y.detach().clone()
        self.seed_force[0] = force.detach().clone()
        self.z_samples[0] = out[4]

        if self.use_buffer:
            self.z_buff.push(out[4])
            self.init[0] = True
        else:
            self.init[0] = True
        # print(np.all([orig == chunk for orig,chunk in zip(self.decode.state_dict() , self.chunk_decode.decode.state_dict())])) # sanity check
        return out

    @torch.no_grad()
    def pdf(self, samples):
        xr = torch.as_tensor(samples) #.to(device=self.device,dtype=self.dtype)
        var_data = self.pdf_torch(xr)
        return var_data.cpu().numpy()

    def pdf_torch(self, samples):
        samples = samples.to(device=self.device,dtype=self.dtype)
        if not self.init:
            return self.init_uniform_grid(samples)
        else:
            if self.dx: 
                samples = samples - self.seed_x
            if self.use_buffer:
                z_buff = self.z_buff.get_samples()
                z_mem = z_buff.shape[0]
                num_samps = samples.shape[0]
                latent_space = torch.vstack([torch.cat([zs.repeat(samples.shape[0], 1), samples], dim=1) for zs in z_buff])
            else:
                ## use z_samples (original)
                z_samples = self.z_samples.repeat(samples.shape[0], 1)
                latent_space = torch.cat([z_samples, samples], dim=1)
            if self.use_chunk_decode:
                y_out = self.chunk_decode(latent_space)
            else:
                y_out = self.decode(latent_space)
            y_logvar = y_out[:,:self.ylogvar_dim]  
            var_data = torch.clamp(y_logvar, *self.logvar_lims)
            if self.use_buffer:
                var_data = var_data.reshape(z_mem,num_samps,self.ylogvar_dim)
                var_data = torch.mean(var_data,0)
            var_data = torch.exp(var_data)
            # var_data = var_data / torch.amax(var_data,0)
            # var_data = torch.mean(var_data,1)
            var_data = torch.amax(var_data,1)
            # var_data = var_data / torch.amax(var_data)
            # var_data = torch.clamp(var_data,min=1e-6)
            return var_data.squeeze()

class chunk_decode(torch.nn.Module):
    def __init__(self,decoder,num_threads=20):
        super(chunk_decode,self).__init__()
        self.decode = decoder
        self.num_threads = num_threads

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        fut = [torch.jit.fork(self.decode, ls) for ls in x.chunk(self.num_threads,dim=0)]
        xs = [torch.jit.wait(f) for f in fut]
        return torch.vstack(xs)

from typing import Dict,Tuple,Optional,List
class chunk_forward(torch.nn.Module):
    def __init__(self,model,num_threads=4):
        super(chunk_forward,self).__init__()
        self.model = model
        self.num_threads = num_threads

    def forward(self, x:torch.Tensor, y:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        fut = [torch.jit.fork(self.model,xs,ys,xs) for xs,ys in zip(x.chunk(self.num_threads,dim=0),y.chunk(self.num_threads,dim=0))]
        out = [torch.jit.wait(f) for f in fut]
        img_pred: List[torch.Tensor] = [results[0] for results in out]
        img_logvar: List[torch.Tensor] = [results[1] for results in out]
        z_mu: List[torch.Tensor] = [results[2] for results in out]
        z_logvar: List[torch.Tensor] = [results[3] for results in out]
        z_samples: List[torch.Tensor] = [results[4] for results in out]
        return torch.vstack(img_pred), torch.vstack(img_logvar), torch.vstack(z_mu), torch.vstack(z_logvar), torch.vstack(z_samples)

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    torch.manual_seed(1)
    np.random.seed(1)

    img_dim = [3,45,45] #[3,38,38] # [3,75,75]

    vae = VAE(img_dim=img_dim, z_dim=6, s_dim=2, hidden_dim=[32,16])
    img = np.random.randn(*img_dim)
    print('img.shape:', img.shape)
    plt.imshow(img.T)
    plt.show(block=False)

    y = torch.FloatTensor(img).unsqueeze(0)
    print("y.shape: ", y.shape)
    out = vae.img_encoder(y)
    print("out: ", out.shape)
    x = [0.2,0.2]
    x = torch.FloatTensor(x).unsqueeze(0)
    force = torch.ones(1).unsqueeze(0)
    z_out  = vae.encode(torch.cat([out, force, x], dim=1))
    print(z_out.shape)
    img_pred, img_logvar , z_mu, z_logvar, z_samples, pred_force = vae(x,y,force)
    y_pred = vae.decode(torch.cat([z_samples, x], dim=1))
    print("y_pred shape: ", y_pred.shape)
    print(img_pred.shape, img_logvar.shape, img_logvar[:,0], img_logvar[0,0], img_logvar[0,0].exp().detach().numpy())

    img_new = img_pred.detach().numpy()[0].T
    img_new = np.clip(img_new, 0, 1)

    plt.figure()
    plt.imshow( img_new)

    print('pdf')

    vae.update_dist(x,y,force)
    numpts = 50
    lim = [-1,1]
    x, y = np.meshgrid(np.linspace(lim[0],lim[1],numpts), np.linspace(lim[0],lim[1],numpts))
    samples = np.c_[x.ravel(), y.ravel()]
    out = vae.pdf(samples)
    print(np.min(out),np.max(out))
    plt.show()
