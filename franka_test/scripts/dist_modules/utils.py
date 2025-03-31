#!/usr/bin/env python

########## global imports ##########
import numpy as np
import numpy.random as npr
np.finfo(np.dtype("float32"))
np.finfo(np.dtype("float64"))
import random

import os
import torch
import torch.distributed as dist
try:
    import torch_ccl
except:
    try:
        import oneccl_bindings_for_pytorch
    except:
        pass

import signal
from termcolor import cprint
import time

import psutil
import numa

# --------- yaml tools --------------------

import yaml 
from yaml.resolver import BaseResolver 

def numpy_representer(dumper, data):
    return dumper.represent_scalar(BaseResolver.DEFAULT_SCALAR_TAG,str(data.tolist())) 

def add_yaml_representers():
    yaml.add_representer(np.ndarray,numpy_representer)                                      

# import torch.multiprocessing as mp

# --------- Kill Signal for Processes --------------------
def cleanup():
    try:
        dist.destroy_process_group()
    except:
        # os.system("kill -9 $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}') ")
        pass

class GracefulKiller:
  def __init__(self,rank,world_size):
    self.kill_now = False
    self.rank = rank
    self.world_size = world_size
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self, *args):
    cprint('[KILLER] killing process (RANK {})'.format(self.rank),'red')
    cleanup()
    self.kill_now = True

class SimpleQueueWrapper:
    def __init__(self,queue,many=False):
        self.queue = queue
        self.many = many

    def poll(self, timeout=1):
        if not(self.queue.empty()):
            return True
        else:
            time.sleep(timeout)
            return False

    # def poll(self, timeout=1):
        # start_time = time.time()
        # done = False
        # while not done:
        #     duration = (time.time()-start_time)/60
        #     if not(self.queue.empty()):
        #         return True
        #     elif duration >= timeout:
        #         return False
        #     else:
        #         time.sleep(0.1)

    def recv(self):
        if self.many:
            return self.queue.get_many()
        else:
            return self.queue.get()
    def send(self,x):
        return self.queue.put(x)

    def empty(self):
        return self.queue.empty()

    def qsize(self):
        return self.queue.qsize()

# --------- setup --------------------

def set_seeds(seed):
    random.seed(seed)
    npr.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

def check_numa(cores,numa_num):
    if hasattr(numa,'info'):
        return set(cores).issubset(numa.info.node_to_cpus(numa_num))
    else:
        return set(cores).issubset(numa.node_to_cpus(numa_num))

def get_node_info():
    num_cores = psutil.cpu_count(logical = False)
    if hasattr(numa,'info'):
        return numa.info.get_num_configured_nodes(),num_cores
    else:
        return numa.get_max_node()+1,num_cores

def round_up_to_even(f):
    return np.ceil(f / 2.) * 2

def get_env_info(world_size,num_cores=20,num_cores_extra=2,num_extra_processes=2,start_offset=0,print_debug=False,async_main=False,ccl_worker_count=4):
    if 'OMP_NUM_THREADS' in os.environ.keys() and os.environ['OMP_NUM_THREADS'] is not None:
        num_cores = int(os.environ['OMP_NUM_THREADS'])
    num_nodes,num_real_cores = get_node_info()
    if (num_nodes > 1) and (world_size > 1):
        denom = round_up_to_even(world_size)
    else: 
        denom = world_size
    max_cores_per_proc = int(np.floor((num_real_cores +1 - start_offset - round_up_to_even(round_up_to_even(num_extra_processes+1*async_main)*num_cores_extra) ) / denom))

    if max_cores_per_proc <= 0:
        raise ValueError('not enough cores available for requested configuration')

    if max_cores_per_proc < num_cores:
        num_cores = max_cores_per_proc
    
    if ccl_worker_count > 0:
        if ccl_worker_count > num_real_cores/num_nodes/world_size:
            ccl_worker_count = int(num_real_cores/num_nodes/world_size) 

        if num_nodes == 1: 
            first_ccl_core = num_real_cores*2 - world_size*ccl_worker_count -1
            last_ccl_core = num_real_cores*2 - 1
        elif num_nodes == 2: 
            first_ccl_core = int(num_real_cores*1.5 -  world_size*ccl_worker_count /2 )-1
            last_ccl_core = int(num_real_cores*1.5 + world_size*ccl_worker_count /2)-1

    env_info = {}
    env_info['max_cores_per_proc'] = max_cores_per_proc
    offset = start_offset
    extra_start = 0 
    occupied_cores = []
    if ccl_worker_count > 0 and (num_nodes == 1 or num_nodes == 2):
        occupied_cores = list(range(first_ccl_core,last_ccl_core))
    for rank in range(world_size+num_extra_processes):
        env_info[rank]={}
        if rank >= world_size and num_nodes > 1 and rank == ( world_size + num_extra_processes/num_nodes ):
            extra_start += int(world_size/2*num_cores)
        checked = False
        while not checked:
            if rank < world_size:
                new_cores = list(range(offset+rank*num_cores,offset+(rank+1)*num_cores))
            elif async_main and rank == world_size: 
                new_cores = list(range(extra_start+(rank-world_size)*num_cores_extra,extra_start+(rank-world_size+1)*num_cores_extra*2))
                if hasattr(numa,'memory'):
                    if check_numa(new_cores,0):
                        start_core = num_real_cores
                        end_core =int(num_real_cores+num_real_cores/num_nodes)
                        extra_cores = np.arange(start_core,end_core)
                        no_overlap = np.array([not c in occupied_cores for c in extra_cores])
                        extra_cores = extra_cores[no_overlap].tolist()
                        new_cores = new_cores + extra_cores
                    elif check_numa(new_cores,1):
                        start_core = int(num_real_cores+num_real_cores/num_nodes)
                        end_core = num_real_cores*2
                        extra_cores = np.arange(start_core,end_core)
                        no_overlap = np.array([not c in occupied_cores for c in extra_cores])
                        extra_cores = extra_cores[no_overlap].tolist()
                        new_cores = new_cores + extra_cores
            else:
                new_cores = list(range(extra_start+(rank-world_size)*num_cores_extra,extra_start+(rank-world_size+1)*num_cores_extra))
            if check_numa(new_cores,0) or check_numa(new_cores,1):
                checked = True
            else:
                new_offset = np.where( np.array(new_cores) >= num_real_cores/num_nodes)[0][0]
                offset += new_offset
                if new_offset >= (num_cores_extra*num_extra_processes)/num_nodes:
                    extra_start = new_cores[0]
                else:
                    extra_start += new_offset
            if all([c in occupied_cores for c in new_cores]):
                checked = False
                extra_start += len(new_cores)
            elif any([c in occupied_cores for c in new_cores]):
                checked = False
                extra_start += 1
        occupied_cores = occupied_cores + new_cores
        env_info[rank]['OMP_NUM_THREADS'] = str(len(new_cores))
        affinity = str(new_cores).replace('[','{').replace(']','}')
        env_info[rank]['affinity'] = affinity
    if ccl_worker_count > 0:
        # specify communication cores
        # if offset == 0:
            # first_ccl_core = int(world_size*num_cores+num_cores_extra*num_extra_processes)
        # else:
            # first_ccl_core = int(offset+world_size*num_cores)

        # first_ccl_core = new_cores[-1]+1
        # last_ccl_core = num_real_cores*2-1  # logical cores are 80-159
        if num_nodes == 1 or num_nodes == 2:
            os.environ['CCL_WORKER_AFFINITY'] = f"{first_ccl_core}-{last_ccl_core}" # "auto"
            os.environ['CCL_WORKER_COUNT'] = f"{ccl_worker_count}"  # number of worker threads per rank (default = 1)
        else:
            os.environ['CCL_WORKER_AFFINITY'] = "auto"
        os.environ["CCL_ATL_TRANSPORT"] = "ofi"
        if print_debug:
            cprint(f"new comm affinity {os.environ['CCL_WORKER_AFFINITY']}",'yellow')
    return env_info

def reset_env(): 
    for attr in ['KMP_AFFINITY','CCL_WORKER_AFFINITY','CCL_WORKER_COUNT','CCL_ATL_TRANSPORT']:
        os.environ[attr] = ""
    num_nodes,num_real_cores = get_node_info()
    os.environ['OMP_NUM_THREADS'] = f"{int(num_real_cores/num_nodes)}"
        
def set_env(rank,world_size,env_info,num_cores=20,num_cores_extra=2,set_cpu_affinity=False,print_debug=False):
    os.environ['OMP_NUM_THREADS'] = env_info[rank]['OMP_NUM_THREADS']
    if set_cpu_affinity: # main
        num_cores = np.fromstring(env_info[rank]['affinity'][1:-1],sep=', ',dtype=int).tolist()
        p = psutil.Process()
        p.cpu_affinity(num_cores)
        if print_debug:
            cprint(f"[RANK {rank}] pid {p.pid} comp affinity {env_info[rank]['affinity']}",'red')
    else:
        os.environ['KMP_AFFINITY']=f"granularity=fine,proclist=[{env_info[rank]['affinity']}],explicit"

def get_num_cores():
    p = psutil.Process()
    return len(p.cpu_affinity())

def setup(rank,world_size,seed,use_gpu=False,skip_numa=False,print_debug=False):
    p = psutil.Process()
    cores = p.cpu_affinity()
    # numa.info.numa_hardware_info()['node_cpu_info']
    if not skip_numa and hasattr(numa,'memory'):
        if check_numa(cores,0):
            numa.memory.set_membind_nodes(0)
            numa_bind = 0
        elif check_numa(cores,1):
            numa.memory.set_membind_nodes(1)
            numa_bind = 1
        else:
            raise ValueError("cores split across numa nodes so don't know which memory to bind to")
    else:
        numa_bind = 0
        print('skipping numa bind')
    if print_debug:
        cprint(f"[RANK {rank}] pid {p.pid} comp affinity {cores} | numa bind {numa_bind}",'red')
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    torch.set_flush_denormal(True)
    if use_gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seeds(seed)
    return GracefulKiller(rank,world_size)

def build_arg_dicts(args,replay_buffer):
    if not('batch_per_proc' in  args.__dict__.keys()):
        if  ('num_update_proc' in  args.__dict__.keys()) :
            args.batch_per_proc = int(args.batch_size/args.num_update_proc)
        else:
            args.batch_per_proc = args.batch_size
            args.num_update_proc = 1
    if not('dx' in  args.__dict__.keys()):
        args.dx = False
    model_dict = {'img_dim':args.image_dim, 'z_dim':args.z_dim, 's_dim':args.s_dim, 'hidden_dim':args.hidden_dim,'y_logvar_dim':args.y_logvar_dim, 'CNNdict':args.CNNdict,'dx':args.dx}
    train_dict = {'args':args,'model_dict':model_dict,'replay_buffer':replay_buffer,'use_gpu':args.use_gpu}
    return model_dict,train_dict
