  // adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/src/cartesian_pose_example_controller.cpp
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

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
#include "franka_test/PoseCmd.h"
#include "print_pose.h"


namespace franka_test {


    class CartesianPoseInterface : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaPoseCartesianInterface>
    {
    private:

        franka_hw::FrankaPoseCartesianInterface*                cartesian_pose_interface_;
        std::unique_ptr<franka_hw::FrankaCartesianPoseHandle>   cartesian_pose_handle_;
        ros::Duration           elapsed_time_;
        std::array<double, 16>  current_pose_{};

        // ros::Subscriber     pose_cmd_sub;
        ros::Subscriber     reset_sub;
        ros::Subscriber     increase_z_sub;
        ros::ServiceServer  service;
        bool                read_message = false;
        // float               decay_rate   = 0.99;
        float               dt           = 0.2;
        bool enforce_dt = false;

        std::array<double, 16> raw_pose_cmd {};
        std::array<double, 16> filtered_raw_pose_cmd {};
        // std::array<double, 16> filtered_target_cmd {};

        float alpha1   = 0.999;
        float alpha2   = 0.99;
        float max_force = 500.0;

        ros::Subscriber     force_sub;
        bool                use_external_fts = false;
        std::array<double, 6> wrench {};

    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh)
        {

            service = nh.advertiseService("/pose_cmd", &CartesianPoseInterface::cmd_callback, this);
            // pose_cmd_sub = nh.subscribe("/pose_cmd", 1, &CartesianPoseInterface::cmd_callback, this);

            cartesian_pose_interface_ = robot_hardware->get<franka_hw::FrankaPoseCartesianInterface>();
            if (cartesian_pose_interface_ == nullptr) {
                ROS_ERROR(
                  "CartesianPoseInterface: Could not get Cartesian Pose "
                  "interface from hardware");
                return false;
            }

            std::string arm_id;
            if (!nh.getParam("arm_id", arm_id)) {
                ROS_ERROR("CartesianPoseInterface: Could not get parameter arm_id");
                return false;
            }

            try {
                cartesian_pose_handle_ = std::make_unique<franka_hw::FrankaCartesianPoseHandle>(
                                        cartesian_pose_interface_->getHandle(arm_id + "_robot")
                                    );
            } catch (const hardware_interface::HardwareInterfaceException& e) {
                ROS_ERROR_STREAM(
                    "CartesianPoseInterface: Exception getting Cartesian handle: " << e.what());
                return false;
            }

            reset_sub = nh.subscribe("/reset_control_commands", 1, &CartesianPoseInterface::resetCallback, this);
            increase_z_sub = nh.subscribe("/increase_z", 1, &CartesianPoseInterface::upCallback, this);
            force_sub = nh.subscribe("/ee_wrench", 1, &CartesianPoseInterface::forceCallback, this);

            return true;
        }

        void starting(const ros::Time&) {
            current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
            filtered_raw_pose_cmd = current_pose_;
            raw_pose_cmd = current_pose_;

            elapsed_time_ = ros::Duration(0.0);
            ros::param::get("/max_force", max_force);
            ros::param::get("/dt", dt);
            ros::param::get("/enforce_dt", enforce_dt);
            ros::param::get("/use_external_fts", use_external_fts);
        }

        void update(const ros::Time&, const ros::Duration& period) {

            // update the current pose
            std::array<double, 16>  current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;

            if (read_message == true) {
                // resetting the duration if there was a message
                elapsed_time_ = ros::Duration(0.);
                read_message = false; // reset the message
            }
            else {
                // updating if there wasn't, there could be something wrong
                elapsed_time_ += period;
                if (enforce_dt && (elapsed_time_.toSec() > dt)) { // stop moving if time is too long
                  raw_pose_cmd = filtered_raw_pose_cmd;
                }
            }


            // check forces
            if (!use_external_fts){
              wrench = cartesian_pose_handle_->getRobotState().O_F_ext_hat_K; // extimated external wrench actinv on stiffness frame relative to base frame
            }
            bool update;
            for (int i=0; i < 16; i++) {
              update = true;
              if ((i > 11) && (i < 15)){ // check wrench for translation dims only
                if ((abs(wrench[i-12]) > max_force) && (signbit(wrench[i-12]) == signbit(raw_pose_cmd[i]))) {
                  update = false; // don't update
                  ROS_INFO_STREAM("idx " << i << " exceeded max force " << max_force);
                  raw_pose_cmd[i] = current_pose_[i];
                  raw_pose_cmd[14] += 0.01; // z
                }
              }
              if (update) {
                filtered_raw_pose_cmd[i] = alpha1 * filtered_raw_pose_cmd[i] + (1.0-alpha1) * raw_pose_cmd[i];
                current_pose_[i] = alpha2 * current_pose_[i] + (1.0-alpha2) * filtered_raw_pose_cmd[i];
              }
            }

            cartesian_pose_handle_->setCommand(current_pose_);

        }

        bool cmd_callback(franka_test::PoseCmd::Request &req,
                          franka_test::PoseCmd::Response &res) {

            tf2::Quaternion q;
            tf2::fromMsg(req.pose_cmd.rotation, q);
            // q.setRPY(req.pose_cmd.data[3],req.pose_cmd.data[4],req.pose_cmd.data[5]);
            tf2::Matrix3x3 m = tf2::Matrix3x3(q);
            // Pose is represented as a 4x4 matrix in column-major format.
            for (int i = 0; i<3 ; i++) {
              tf2::Vector3 v = m.getColumn(i);
              raw_pose_cmd[i*4] = v.x();
              raw_pose_cmd[i*4+1] = v.y();
              raw_pose_cmd[i*4+2] = v.z();
              raw_pose_cmd[i*4+3] = 0;
              // x y z are 12, 13, 14
              // raw_pose_cmd[12+i] = req.pose_cmd.data[i];
            }


            // double dist = (std::pow(req.pose_cmd.translation.x,2) + std::pow(req.pose_cmd.translation.y,2));

            raw_pose_cmd[12] = req.pose_cmd.translation.x;
            raw_pose_cmd[13] = req.pose_cmd.translation.y;
            raw_pose_cmd[14] = req.pose_cmd.translation.z;
            raw_pose_cmd[15] = 1;
            read_message = true;

            res.pose = print_pose(raw_pose_cmd);
            res.success = cartesian_pose_handle_->getRobotState().control_command_success_rate > 0.5;
            if (!res.success){
              current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
              filtered_raw_pose_cmd = current_pose_;
            }
            return true;
        }

        void resetCallback(const std_msgs::Empty::ConstPtr& msg){
          current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
          // filtered_raw_pose_cmd = current_pose_;
          raw_pose_cmd = current_pose_;
          // raw_pose_cmd = filtered_raw_pose_cmd;

          elapsed_time_ = ros::Duration(0.0);
        }

        void upCallback(const std_msgs::Empty::ConstPtr& msg){
          raw_pose_cmd = filtered_raw_pose_cmd;
          raw_pose_cmd[14] += 0.01; // z
        }

        void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg){
          wrench = {{msg->wrench.force.x,msg->wrench.force.y,msg->wrench.force.z,
            msg->wrench.torque.x,msg->wrench.torque.y,msg->wrench.torque.z}};
        }
};

}  // namespace franka_test

PLUGINLIB_EXPORT_CLASS(franka_test::CartesianPoseInterface,
                       controller_interface::ControllerBase)
