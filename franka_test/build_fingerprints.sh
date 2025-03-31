#!/bin/sh

flag=${1-False}
cond=${2-none}
eval_mod=${3-''}

source test_env_vars.sh $cond

test_path="$data_path/$sensor_method$sensor_mod/${explr_method}_${padded_seed}${end_path}"
model_path="model_${model_string}.pth"

tmp_buffer_name=${buffer_name}

echo $test_path 
if [ $pybullet != True -a $bias_fts == True ]; then rosservice call /Bias_sensor "{}"; fi
if [ $pybullet != True ]; then rostopic pub -1 /reset_joints std_msgs/Empty "{}" ; sleep 2;  fi # reset joints 
if [ $pybullet != True ]; then rostopic pub -1 /reset std_msgs/Empty "{}"; sleep 2;  fi # reset to start configuration
roslaunch franka_test generate_fingerprints.launch test_method:=$test_method test_path:=$test_path model_path:=$model_path save_folder:="eval_${model_string}_${explr_states}${eval_mod}/" explr_states:=$explr_states base_fp_name:=$base_fp_name use_gui:=$use_gui tdist_mode:=$tdist_mode pybullet:=$pybullet render_pybullet:=$render_pybullet cam_z:=$starting_height skip:=$skip sensor_method:=$sensor_method buffer_name:=$tmp_buffer_name num_fp_samples:=$num_fp_samples


##### manual
# fingerprint_name=fp_id3
# roslaunch franka_test manual_fingerprint.launch test_method:=$test_method test_path:=$test_path model_path:=$model_path save_folder:="eval_${model_string}_${explr_states}${eval_mod}/" explr_states:=$explr_states fingerprint_name:=$fingerprint_name use_gui:=$use_gui pybullet:=$pybullet render_pybullet:=$render_pybullet cam_z:=$starting_height skip:=$skip sensor_method:=$sensor_method buffer_name:=$tmp_buffer_name
