#!/bin/sh

cond=${1-none}

################ ROSLAUNCH params ################

pybullet=True
render_pybullet=True
use_gui=True
clustering=True
starting_height=0.3
test_config_file=test_config_pybullet.yaml # test_config.yaml # 
bias_fts=False # only applicable if you have an external force torque sensor

if [ $cond != 'batch' ]
then
  seed=0
  explr_method="entklerg" #"entklerg" "randomWalk" "unifklerg"
  sensor_method="rgb" # "rgb" "intensity"
else 
  echo 'got batch flag---not setting seed, explr_method, or sensor_method'
fi
explr_states="xyw"
sensor_mod="/demo"
end_path="_${explr_states}_two_objects"
distributed=True
ddp=True
async=True
num_trainers=4
data_to_ctrl_rate=1 # e.g. 3 runs data collection 3 times during control step
dt=0.2

## build fingerprints
printf -v padded_seed "%04d" $seed
test_method="explr" # explr, grid, circle
num_model_steps="-1"
num_fp_samples=50
skip=2
base_fp_name="fp_id"
tdist_mode="sphere" # sphere, cone, cylinder, uniform

if [ ${pybullet} == True ]
then
  data_path="sim_data"
else 
  data_path="data"
fi

if [ ${num_model_steps} == -1 ]
then
  model_string="postexplr"
#   model_string="final"
else 
  model_string="final_${num_model_steps}steps"
fi

## test fingerprints
num_id_steps=1000
fp_test_config_file="fp_trainer_config.yaml"
belief_plotting_rate=25
save_name="learning${end_path}"
if [ $cond != 'batch' ]
then
  fingerprint_names="fp_id0 fp_id1" # fp_id2" ## starts with base_fp_name
fi
explr_methods="$explr_method "
seeds="$padded_seed "

## capture workspace
square=False
# save_name="v2_${save_name}"
