#!/bin/sh

cond=${1-none}

source test_env_vars.sh $cond

test_path="$data_path/$sensor_method$sensor_mod/${explr_method}_${padded_seed}${end_path}"

if [ ${square} == True ]
then
  save_name="../${save_name}_square_"
else
  save_name="../${save_name}_"
fi

roslaunch franka_test capture_ws.launch test_path:=$test_path save_name:=$save_name square:=$square pybullet:=$pybullet render_pybullet:=$render_pybullet cam_z:=$starting_height use_gui:=$use_gui
