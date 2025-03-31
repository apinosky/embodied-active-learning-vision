// adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/include/franka_example_controllers/cartesian_velocity_example_controller.h
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#pragma once

#include <cmath>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <controller_interface/multi_interface_controller.h>
#include <franka_hw/franka_cartesian_command_interface.h>

#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/robot_hw.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <std_msgs/Empty.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/WrenchStamped.h>
#include "franka_test/VelCmd.h"

namespace franka_test {


    class CartesianVelInterface : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaVelocityCartesianInterface>
    {
    private:

        franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_;
        std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;
        ros::Duration           elapsed_time_;
        std::array<double, 16>  current_pose_{};

        ros::Subscriber     reset_sub;
        ros::ServiceServer  service;
        bool                read_message = false;
        // float               decay_rate   = 0.99;
        float               dt;
        float               max_force;

        std::array<double, 6> raw_cmd = {{0., 0., 0., 0., 0., 0.}};
        std::array<double, 6> filtered_cmd = {{0., 0., 0., 0., 0., 0.}};

        float alpha   = 0.995;

        ros::Subscriber     force_sub;
        bool                use_external_fts = false;
        std::array<double, 6> wrench {};


    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh) override;

        void starting(const ros::Time&) override;

        void update(const ros::Time&, const ros::Duration& period)  override;

        bool cmd_callback(franka_test::VelCmd::Request &req,
                          franka_test::VelCmd::Response &res);

        void resetCallback(const std_msgs::Empty::ConstPtr&);

        void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr&);

    };
}  // namespace franka_test
