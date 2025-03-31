// adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/include/franka_example_controllers/joint_position_example_controller.h
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#pragma once

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
        double               dt;
        bool                enforce_dt;

        std::array<double, 7> raw_pose_cmd {};
        std::array<double, 7> filtered_raw_pose_cmd {};

        double Kp   = 0.1;
        double Ki   = 0.01;
        double Kd   = 0.5;
        std::array<double, 7> pre_error {};
        std::array<double, 7> integral {};

    public:
    
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh) override;

        void starting(const ros::Time&)  override;

        void update(const ros::Time&, const ros::Duration& period)  override;

        bool cmd_callback(franka_test::JointCmd::Request &req,
                          franka_test::JointCmd::Response &res);


    };
}  // namespace franka_test
