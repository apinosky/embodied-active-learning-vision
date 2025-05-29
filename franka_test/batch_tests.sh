#!/bin/sh

source setup.bash 
source test_env_vars.sh batch

train=True
fp=False
test=False

eval_mod=''

num_seeds=0

explr_methods='entklerg '

for seed in $(seq 0 $num_seeds)
do
    for explr_method in $explr_methods #  randomWalk unifklerg uniform entklerg 
    do
        for sensor_method in rgb # rgb intensity
        do
            echo $explr_method $sensor_method $seed
            if [ $train = True ]
            then 
                if [ $pybullet != True ]; then rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"; fi; 
                if [ $pybullet != True ]; then rostopic pub -1 /reset_joints std_msgs/Empty "{}"; sleep 2; fi; # reset joints 
                source run.sh batch
            fi 
            if [ $fp = True ]
            then
                source build_fingerprints.sh online batch $eval_mod
            fi
        done
    done
done

# test
if [ $test = True ]
then
    if [ $pybullet != True ]; then rostopic pub -1 /franka_control/error_recovery/goal franka_msgs/ErrorRecoveryActionGoal "{}"; fi; 
    if [ $pybullet != True ]; then rostopic pub -1 /reset_joints std_msgs/Empty "{}"; sleep 2;  fi; # reset joints 
    if [ $pybullet != True ]; then rostopic pub -1 /reset std_msgs/Empty "{}"; sleep 2;  fi; # reset to start configuration
    source test_fingerprints_auto_batch.sh batch $eval_mod
    # source test_fingerprints_auto.sh batch $eval_mod
fi
