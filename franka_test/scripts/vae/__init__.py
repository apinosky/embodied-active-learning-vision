#!/usr/bin/env python

def get_VAE(learn_force): 
    if learn_force:
        from .vae_force import VAE
    else:
        from .vae import VAE
    return VAE
