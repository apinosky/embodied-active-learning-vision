#!usr/bin/env python

### Python Imports ###
import numpy as np
# import numpy.random as npr
# from scipy.stats import norm

def psi_fn(traj, samples, std):
    inner = np.square(traj-samples)/std
    psi = np.exp(-0.5 * np.sum(inner, 2))
    return psi

def kldiv_grad_vec(x,samples, explr_idx, std, importance_ratio, nu, dtype):
    """ gradient of state footprint; grad is d_psi/d_x; psi is the part inside the integral of q"""
    grad = np.zeros((samples.shape[0],x.shape[0]),dtype=dtype)
    diff = -(x[explr_idx]-samples)/np.abs(std)
    psi = psi_fn(x[None,None,explr_idx],samples[:,None,:], np.abs(std), dtype)
    grad[:,explr_idx] = diff*psi
    grad = np.sum(grad*importance_ratio[:,None],axis=0)
    return grad # / nu

def traj_footprint_vec(traj, samples, explr_idx, std, nu, dtype=None):
    """ time-averaged footprint of trajectory (used to calc q)"""
    psi = psi_fn(traj[None,:,explr_idx],samples[:,None,:],np.abs(std))
    pdf = np.sum(psi, -1)
    return pdf / nu

def traj_spread_vec(traj, samples, explr_idx, std, nu, dtype=None):
    """ time-averaged footprint of trajectory (used to calc q)"""
    psi = psi_fn(traj[None,:,explr_idx],samples[:,None,:],np.abs(std))
    pdf = np.amax(psi, -1)
    return pdf 
def traj_footprint(traj, s, explr_idx, std, nu):
    """ time-averaged footprint of trajectory (used to calc q)"""
    psi = psi_fn(traj[None,:,explr_idx],s[None,None,:],np.abs(std))
    pdf = np.sum(psi, 0)
    # pdf = np.mean(psi, 0)  ### don't use this if incrementally updating q
    pdf = np.clip(pdf,1e-6,None)
    return pdf / nu

def renormalize(dist,dim=None,min_val=1e-6):
    if dim is not None:
        dist /= np.sum(dist,dim,keepdims=True)
        dist = np.clip(dist,min_val,None)
        dist = np.log(dist)
        dist -= np.max(dist,dim,keepdims=True)
        dist = np.exp(dist)
    else:
        dist /= np.sum(dist)
        dist = np.clip(dist,min_val,None)
        dist = np.log(dist)
        dist -= np.max(dist)
        dist = np.exp(dist)
    return dist

def cost_norm(dist): 
    dist[np.isnan(dist)] = 1e-6
    dist /= np.sum(dist)
    # dist = np.clip(dist,1e-6,None)
    return dist

######### jit functions #########

try:
    import numba as nb
    parallel = False
    @nb.jit(nopython=True,fastmath=True, cache=True, parallel=parallel)
    def loopy_kldiv_grad_vec(x,samples,std,importance_ratio,outer_loops,dims,dtype):
        grad = np.zeros((dims,outer_loops),dtype=dtype)
        for outer_idx in nb.prange(outer_loops):
            diff = (x-samples[outer_idx])
            grad[:,outer_idx] = -(diff/std)
            tmp = np.square(diff)/std
            for dim_idx in range(1,dims):
                tmp[0] += tmp[dim_idx]
            grad[:,outer_idx] *= np.exp(-0.5 * tmp[0]) # psi
        out = np.zeros(dims,dtype=dtype)
        for idx in nb.prange(outer_loops):
            out += grad[:,idx]*importance_ratio[idx]
        return out  #/outer_loops

    def kldiv_grad_vec_numba(x,samples, explr_idx, std, importance_ratio, nu, dtype):
        """ gradient of state footprint """
        grad = np.zeros(x.shape[0],dtype=dtype)
        x_tmp = x[explr_idx]
        outer_loops,dims = samples.shape
        grad[:dims] = loopy_kldiv_grad_vec(x_tmp,samples,std,importance_ratio,outer_loops,dims,dtype)
        return grad # / nu

    @nb.jit(nopython=True,fastmath=True, cache=True, parallel=parallel)
    def loopy_traj_footprint_vec(traj,samples,std,outer_loops,inner_loops,dims,dtype):
        pdf = np.zeros(outer_loops,dtype=dtype)
        for outer_idx in nb.prange(outer_loops):
            inner = np.square(traj-samples[outer_idx])/std
            for inner_idx in nb.prange(inner_loops):
                for dim_idx in nb.prange(1,dims):
                    inner[inner_idx,0] += inner[inner_idx,dim_idx]
                psi = np.exp(-0.5*inner[inner_idx,0])
                pdf[outer_idx] += psi
        return pdf # / inner_loops

    def traj_footprint_vec_numba(traj, samples, explr_idx, std, nu,dtype):
        tmp_traj = traj[:,explr_idx]
        abs_std = np.abs(std)
        inner_loops = traj.shape[0]
        outer_loops,dims = samples.shape
        return loopy_traj_footprint_vec(tmp_traj,samples,abs_std,outer_loops,inner_loops,dims,dtype) / nu
    
except: 
    print("numba import error, not using numba. (you can manually disable numba by setting `set use_numba=False` in klerg.py")
    kldiv_grad_vec_numba = kldiv_grad_vec
    traj_footprint_vec_numba = traj_footprint_vec
    
