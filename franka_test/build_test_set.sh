#!/bin/sh

source test_env_vars.sh

num_steps=20

loc="bottom_left" 
name="duck"

# loc="top_right"
# name="plant"

test_path="$data_path/$sensor_method$sensor_mod/${explr_method}_${padded_seed}${end_path}"
save_folder="/" 

echo $test_path
# rostopic pub -1 /reset std_msgs/Empty "{}" # reset to start configuration
roslaunch franka_test build_test_set.launch test_path:=$test_path save_folder:=$save_folder pybullet:=$pybullet render_pybullet:=$render_pybullet cam_z:=$starting_height loc:=$loc fingerprint_name:=$name dt:=$dt test_config_file:=$test_config_file sensor_method:=$sensor_method num_steps:=$num_steps