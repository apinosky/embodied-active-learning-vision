  // adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/src/joint_position_example_controller.cpp
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#include <cmath>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>

#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/robot_hw.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include "franka_test/JointCmd.h"


namespace franka_test {

    class JointPositionInterface : public controller_interface::MultiInterfaceController<
                                            hardware_interface::PositionJointInterface>
    {
    private:

        hardware_interface::PositionJointInterface* position_joint_interface_;
        std::vector<hardware_interface::JointHandle> position_joint_handles_;
        ros::Duration           elapsed_time_;

        ros::ServiceServer  service;
        bool                read_message = false;
        double               dt           = 0.2;
        bool enforce_dt = false;

        std::array<double, 7> raw_pose_cmd {};
        std::array<double, 7> filtered_raw_pose_cmd {};

        double Kp   = 0.02;
        double Ki   = 0.;
        double Kd   = 0.02;
        std::array<double, 7> pre_error {};
        std::array<double, 7> integral {};


    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh)
        {

            service = nh.advertiseService("/joint_cmd", &JointPositionInterface::cmd_callback, this);

            position_joint_interface_ = robot_hardware->get<hardware_interface::PositionJointInterface>();
            if (position_joint_interface_ == nullptr) {
              ROS_ERROR(
                  "JointPositionInterface: Error getting position joint interface from hardware!");
              return false;
            }
            std::vector<std::string> joint_names;
            if (!nh.getParam("joint_names", joint_names)) {
              ROS_ERROR("JointPositionInterface: Could not parse joint names");
            }
            if (joint_names.size() != 7) {
              ROS_ERROR_STREAM("JointPositionInterface: Wrong number of joint names, got "
                              << joint_names.size() << " instead of 7 names!");
              return false;
            }
            position_joint_handles_.resize(7);
            for (size_t i = 0; i < 7; ++i) {
              try {
                position_joint_handles_[i] = position_joint_interface_->getHandle(joint_names[i]);
              } catch (const hardware_interface::HardwareInterfaceException& e) {
                ROS_ERROR_STREAM(
                    "JointPositionInterface: Exception getting joint handles: " << e.what());
                return false;
              }
            }

            return true;
        }

        void starting(const ros::Time&) {
            elapsed_time_ = ros::Duration(0.0);
            for (size_t i = 0; i < 7; ++i) {
                // init commands
                pre_error[i] = 0.;
                integral[i] = 0.;
                raw_pose_cmd[i] = position_joint_handles_[i].getPosition();
                filtered_raw_pose_cmd[i] = position_joint_handles_[i].getPosition();
            }

        }

        void update(const ros::Time&, const ros::Duration& period) {


            if (read_message == true) {
                // resetting the duration if there was a message
                elapsed_time_ = ros::Duration(0.);
                read_message = false; // reset the message
            }
            else {
                // updating if there wasn't, there could be something wrong
                elapsed_time_ += period;
                if (enforce_dt && (elapsed_time_.toSec() > dt*5.)) { // stop moving if time is too long
                  raw_pose_cmd = filtered_raw_pose_cmd;
                }
            }

            for (size_t i = 0; i < 7; ++i) {
              double current_pos = position_joint_handles_[i].getPosition();
              //PID
              double error = raw_pose_cmd[i] - current_pos;
              integral[i] += error * dt;
              double derivative = (error - pre_error[i]) / dt;
              pre_error[i] = error;
              double delta = Kp*error + Ki*integral[i] + Kd*derivative;

              // ROS_INFO_STREAM("JointPositionInterface: curr des delta "
              //                 << current_pos << " " << raw_pose_cmd[i] << " " << delta << " ");

              double max_ang = 2.0/1000.;
              if (abs(delta) > max_ang){
                if (delta > 0.){
                  delta = max_ang;
                }
                else {
                  delta = - max_ang;
                }
              }
              // else if (abs(delta) < 0.01){
              //     delta = 0.;
              // }
              filtered_raw_pose_cmd[i] = current_pos + delta;
              position_joint_handles_[i].setCommand(filtered_raw_pose_cmd[i]);
            }

        }

        bool cmd_callback(franka_test::JointCmd::Request &req,
                          franka_test::JointCmd::Response &res) {

            for (size_t i = 0; i < 7; ++i) {
              raw_pose_cmd[i] = req.joint_cmd[i];
              filtered_raw_pose_cmd[i] = position_joint_handles_[i].getPosition();
            }
            read_message = true;

            res.success = true;
            return true;
        }

};

}  // namespace franka_test

PLUGINLIB_EXPORT_CLASS(franka_test::JointPositionInterface,
                       controller_interface::ControllerBase)
