#!/bin/sh

cond=${1-none}

source test_env_vars.sh $cond

if [ $pybullet != True -a $bias_fts == True ]; then rosservice call /Bias_sensor "{}"; fi
roslaunch franka_test run.launch pybullet:=$pybullet seed:=$seed explr_method:=$explr_method sensor_method:=$sensor_method sensor_mod:=$sensor_mod path_mod:=$end_path distributed:=$distributed ddp:=$ddp async:=$async num_trainers:=$num_trainers render_pybullet:=$render_pybullet cam_z:=$starting_height use_gui:=$use_gui clustering:=$clustering dt:=$dt data_to_ctrl_rate:=$data_to_ctrl_rate test_config_file:=$test_config_file explr_states:=$explr_states
