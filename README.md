# Franka experiments to replicate experiments for "Embodied Active Learning of Generative Sensor-Object Models"

- All hardware for experiments run with ROS Melodic and Python 2 on a Franka Panda robot.
- All calculations for experiments run with ROS Noetic and Python 3.

Note: all experiments can be run on a single computer if desired.

## Overview

This directory contains the code for learning the RGB camera sensory model in a hardware or with a PyBullet simulator, where an RGB camera is attached to the end-effector of the Franka Panda robot in an environment with two objects, a rubber duck and a plant.

## Setup


1. Clone git repository or unzip source files into a catkin workspace (recommended location `$HOME/franka_ws/src`)

2. Initialize workspace by running `catkin_make` from `franka_ws` directory

Note 1: you may get some errors the first time since you haven't run rosdep yet. If so, run the next line then run this again.

Note 2: any time you add/remove files or edit C++ files, it's good practice to rerun `catkin_make`

3. Install ROS dependencies with `rosdep install franka_test`

4. Install python dependencies by running `pip install -r requirements.txt`  from the `franka_test` directory

5. If you want to run distributed training and/or use Intel's pytorch bindings, install optional python dependencies by running `pip install -r optional-requirements.txt`  from the `franka_test` directory.

6. Source catkin workspace with `source devel/setup.bash`

7. If running on hardware, also be sure to set the `ROS_MASTER_URI` either on the command line or by sourcing `source robot.bash`

## Running the code

For most files, both a launch file and a bash script are provided. Test parameters are set in `config/test_config.yaml` as well as in `test_env_vars.sh`.

### Main scripts (in sequential order):

All launch files can be run with the command  `roslaunch fanka_test <name>.launch`

All source files can be run with the command `source <name>.sh`

| Description | Launch Files | Notes |
| -- | -- | -- |
| Setup hardware and visualizations | `start_robot.launch` | Optional for PyBullet simulation |
| Learn sensor-object model | `run.launch`| Recommended Configuration below |
| Generate fingerprints |  `generate_fingerprints.launch` | alternate: `build_fingerprints.sh` |
| Explore with learned fingerprints |  `test_fingerprints.launch` | alternate: `test_fingerprints.sh` |
| Capture images at each identified object location |  `capture_fingerprint_belief.launch` | alternate: `capture_fingerprints.sh` |
| Capture overhead view of test workspace |  `capture_ws.launch` | alternate: `capture_ws.sh` |

### Recommended configuration for learning sensor-object model:

| parameter | value | notes |
| -- | -- | -- |
| distributed | TRUE | only runs plotters in separate threads|
| ddp | TRUE | if you don't want to distribute the training, you can also set num_trainers=1|

### Configuration-specific hard-coded references (mostly relevant for hardware testing):

| variable | notes | files |
| -- | -- | -- |
| robot_ip | ip address of your robot | `start_robot.launch`; `franka_control.launch`|
| video device | /dev/video* where your webcam is mounted | `run.launch`; `start_robot.launch` |
| ROS_MASTER_URI | ip address of your robot | `robot.bash` |
| robot initial position | the initial / reset pose of the robot is hard coded. must change manually if you want a new start pose for either hardware or simulation | `scripts/go_vel`; `scripts/pybullet_service` |


## More Details

High level description of files included in this directory:

```bash
.
├── config
│   ├── cam.rviz                            # RVIZ hardware visualization
│   ├── fp_trainer_config.yaml              # specify test configuration for fingerprinting
│   ├── franka_control.yaml                 # ROS hardware controller setup file
│   ├── gui.png                             # GUI display
│   ├── panda_arm.urdf.xacro                # modified to add table
│   └── test_config*.yaml                   # test configuration parameters
├── include                                 # C++ ROS Controllers
│   └── file names excluded, see directory
├── launch
│   ├── build_test_set.launch               # collects test sets for later testing over pre-specified locations
│   ├── capture_fingerprint_belief.launch   # capture views of each (potential) identified fingerprint
│   ├── capture_ws.launch                   # capture overhead view of workspace
│   ├── franka_control.launch               # sets up franka control configuration
│   ├── generate_fingerprints.launch        # extract fingerprints and locations from learned model
│   ├── manual_fingerprint.launch           # generate fingerprint at manually specified locations
│   ├── run.launch                          # run model learning experiment
│   ├── start_camera.launch                 # starts webcam interfac
│   ├── start_robot.launch                  # sets up hardware interface
│   └── test_fingerprint.launch             # run identification experiment with specified fingerprints  
├── msg                                     # ROS message files
│   └── file names excluded, see directory
├── scripts
│   ├── control                             # KL ergodic controller
│   │   ├── barrier.py                      # workspace barrier functions for controller
│   │   ├── default_policies.py             # default control policies for KL ergodic controller
│   │   ├── dummy_robot.py                  # base class for random control
│   │   ├── dynamics.py                     # dynamics functions
│   │   ├── klerg.py                        # base class for ergodic control
│   │   ├── klerg_utils.py                  # helper functions for ergodic control
│   │   ├── memory_buffer.py                # stores trajectory history data
│   │   └── robot_config.yaml               # sets default control parameters
│   ├── control_torch                       # KL ergodic controller implemented in pytorch
│   │   ├── barrier.py                      # workspace barrier functions for controller
│   │   ├── default_policies.py             # default control policies for KL ergodic controller
│   │   ├── dynamics.py                     # dynamics functions
│   │   ├── klerg.py                        # base class for ergodic control
│   │   ├── klerg_utils.py                  # helper functions for ergodic control
│   │   ├── memory_buffer.py                # stores trajectory history data
│   │   ├── robot_config*.yaml              # sets default control parameters
│   │   └── rotations.py                    # helper functions for rotation matrices
│   ├── dist_modules
|   │   ├── clustering.py                   # clustering of conditional entropy 
|   │   ├── fingerprint_builder.py          # extract fingerprints (base functions)
|   │   ├── fingerprint_module.py           # test fingerprints (base functions)
│   │   ├── main_async.py                   # recommended main -- runs plotter in a separate process (with option for multiple trainers in separate processes and asynchronous exploration node)
│   │   ├── main_sync.py                    # runs plotter in a separate process (with option for multiple trainers in separate processes)
│   │   ├── sensor_main_module.py           # base model learning sensor interface
│   │   ├── sensor_test_module.py           # base test sensor interface (no learning)
│   │   ├── sensor_utils.py                 # base ros interface for all sensor-based experiments (base class in sensor_main, dist_modules/sensor_main_module, and dist_modules/sensor_main_module); also includes fixed trajectory generators
│   │   ├── test_fingerprint_main.py        # base module for distributed fingerprint testing (fingerprint_mp)
│   │   ├── trainer_ddp.py                  # runs trainer(s) when running in distributed mode
│   │   ├── trainer_module.py               # base trainer -- loaded by main_sync (when not using ddp_trainer) and trainer_ddp
│   │   └── utils.py                        # helper functions for distributed processes (setup/cleanup/seeds/etc.) 
│   ├── franka
│   │   ├── franka_env.py                   # PyBullet simulation
│   │   ├── franka_utils.py                 # functions to transform between tray and robot workspaces
│   │   └── urdf                            # directory with all files for rendering in PyBullet
│   │       └── file names excluded, see directory
│   ├── plotting                            # plotting files
│   │       └── file names excluded, see directory
│   ├── vae
│   │   ├── vae_buffer.py                   # stores training data for CVAE model
│   │   ├── vae_force.py                    # adds force as a sensor (input and output of CVAE)
│   │   ├── vae_utils.py                    # CVAE helper functions
│   │   └── vae.py                          # CVAE RGB model
│   ├── build_fingerprints                  # extract fingerprints and locations from learned model
│   ├── build_manual_fingerprints           # generate fingerprint at manually specified locations
│   ├── capture_fingerprint_belief          # capture views of each (potential) identified fingerprint
│   ├── capture_ws                          # capture overhead view of workspace (hardcoded pose)
│   ├── clustering                          # plots clustering of conditional entropy in real-time
│   ├── dummy_service                       # fake service to debug hardware
│   ├── fingerprint_mp                      # sets up multiprocessing for testing fingerprints
│   ├── go_vel                              # connection between experiment commands and C++ controllers
│   ├── gui                                 # GUI for common ROS commands
│   ├── lamp_brightness                     # enables control of external room brightness 
│   ├── load_config                         # loads and processes test configuration
│   ├── pybullet_service                    # connects to PyBullet for testing in simulation
│   ├── random_listener                     # publishes reset commands to prevent 'random' test methods from getting stuck
│   ├── sensor_main                         # wrapper to set all parameters for running everything main learning experiment
│   └── sensor_test_set                     # TBA
├── src                                     # C++ ROS Controllers
│   └── file names excluded, see directory
├── srv                                     # ROS Services
│   └── file names excluded, see directory
├── batch_tests.sh                          # Wrapper to train, collect fingerprints, multiple seeds back-to-back; then single identification run for all trained seeds
├── build_fingerprints.sh                   # Generate fingerprints from learned model
├── build_test_set.sh                       # Collects data prior to training for debugging purposes
├── capture_fingerprints.sh                 # capture views of each (potential) fingerprint (need belief grids)
├── capture_ws.sh                           # capture overhead view of workspace
├── CMakeLists.txt                          # ROS setup file
├── kill.sh                                 # brute force method to kill any processes that did not shut down properly
├── optional_requirements.txt               # optional python packages
├── package.xml                             # ROS setup file
├── plugin.xml                              # ROS hardware controller setup file
├── record.sh                               # playback test and record in PyBullet (or save images to video)
├── requirements.txt                        # packages required to run this package
├── reset.sh                                # recover from franka motion errors (if not using GUI)
├── robot.bash                              # sources robot for harware tests
├── run.sh                                  # run main launch file for learning (run.launch)
├── setup.py                                # allows ROS to process python files to be accessible throughout package
├── test_env_vars.sh                        # sets environment variables to pass to bash files
└── test_fingerprints*.sh                   # Explore with learned fingerprints (with options to auto-detect fingerprints and run batch of seeds)

```

## Notes
The following files have hard-coded flags to restrict how much data is stored. If you want to save all images, set `video=True` in the following files:

```
scripts/dist_modules/main_async.py
scripts/dist_modules/test_fingerprint_main.py
scripts/dist_modules/trainer_ddp.py
```

## References

Camera_sim module adapted from Prabhakar, A. "Mechanical Intelligence for Learning Embodied Sensor-Object Relationships" (2022), GitHub repository, https://github.com/apr600/mechanical-intelligence.git

Franka controller adapted from "ROS integration for Franka Emika research robots", GitHub repository, https://github.com/frankaemika/franka_ros.git

## Copyright and License

The implementations contained herein are copyright (C) 2025 - 2026 by Allison Pinosky and Todd Murphey and are distributed under the terms of the GNU General Public License (GPL) version 3 (or later). Please see the LICENSE for more information.

Contact: MurpheyLab@u.northwestern.edu

Lab Info: Todd D. Murphey https://murpheylab.github.io/ Northwestern University
