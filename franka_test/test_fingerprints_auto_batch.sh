#!/bin/sh

cond=${1-none}
eval_mod=${2-''}

source test_env_vars.sh $cond

model_path=''
fingerprint_path=''
fingerprint_names=''

path_mod='/../'

for explr_method in $explr_methods 
do
    for seed in $(seq 0 $num_seeds)
    do
        printf -v padded_seed "%04d" $seed
        model_path+="${path_mod}/${explr_method}_${padded_seed}${end_path}/model_${model_string}.pth "
        fingerprint_path+="${path_mod}/${explr_method}_${padded_seed}${end_path}/eval_${model_string}_${explr_states}${eval_mod}/ "
        
        test_path="$data_path/$sensor_method$sensor_mod/${explr_method}_${padded_seed}${end_path}/"
        pushd $test_path/eval_${model_string}_${explr_states}${eval_mod}
        for n in $(ls ${base_fp_name}*.pickle)
        do
            fingerprint_names+="${n%_${test_method}*} "
        done
        popd
        fingerprint_names=${fingerprint_names::-1} # remove blanks
        fingerprint_names+=";"

    done
done
fingerprint_names=${fingerprint_names::-1} # remove blanks

echo $model_path
echo $fingerprint_path
echo $explr_states
echo $fingerprint_names

if [ $pybullet != True -a $bias_fts == True ]; then rosservice call /Bias_sensor "{}"; fi
roslaunch franka_test test_fingerprint.launch test_path:=$test_path model_path:="$model_path" fingerprint_path:="$fingerprint_path" num_steps:=$num_id_steps save_name:="$save_name/explr_${num_id_steps}steps" fingerprint_names:="$fingerprint_names" fingerprint_method:=$test_method explr_states:=$explr_states new_model_explr:=True test_config_file:=$fp_test_config_file belief_plotting_rate:=$belief_plotting_rate async:=True use_gui:=$use_gui pybullet:=$pybullet render_pybullet:=$render_pybullet sensor_method:=$sensor_method cam_z:=$starting_height
