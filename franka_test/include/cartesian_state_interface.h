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
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>

#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/robot_hw.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>

namespace franka_test {


    class CartesianStateInterface : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaStateInterface,
                                            franka_hw::FrankaModelInterface>
    {
    private:

        franka_hw::FrankaStateInterface* state_interface_;
        std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;

        franka_hw::FrankaModelInterface* model_interface_;
        std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;

        ros::Timer ee_timer;
        ros::Publisher      ee_pose_pub;
        ros::Publisher      ee_vel_pub;
        ros::Subscriber     force_sub;
        ros::Publisher      force_pub;
        bool                use_external_fts = false;
        std::array<double, 6> filtered_wrench = {};
        float               alpha;
        float               ft_yaw_offset;

    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh) override;
        void starting(const ros::Time&) override;
        void update(const ros::Time&, const ros::Duration& period)  override;
        void timerCallback(const ros::TimerEvent&);
        void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr&);

    };
}  // namespace franka_test
