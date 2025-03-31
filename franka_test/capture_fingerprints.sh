#!/bin/sh

cond=${1-none}

source test_env_vars.sh $cond

model_path="model_${model_string}.pth "
fingerprint_path="eval_${model_string}_${explr_states}/ "

test_path="$data_path/$sensor_method$sensor_mod/${explr_method}_${padded_seed}${end_path}"


for type in "LatentSpace" # "OutputError"
do
  fp_base_name="explr_${num_id_steps}steps_belief_[]_L2_${type}_final.pickle"
  for angle_method in "mean" "max"
  do
    roslaunch franka_test capture_fingerprint_belief.launch test_path:=$test_path model_path:="$model_path" fingerprint_path:="$fingerprint_path" save_folder:=$save_name fp_base_name:=$fp_base_name fingerprint_names:="$fingerprint_names" explr_states:=$explr_states use_gui:=$use_gui angle_method:=$angle_method pybullet:=$pybullet render_pybullet:=$render_pybullet cam_z:=$starting_height sensor_method:=$sensor_method
  done
done
