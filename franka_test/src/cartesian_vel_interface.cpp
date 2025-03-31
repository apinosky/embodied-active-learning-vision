// adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/src/cartesian_velocity_example_controller.cpp
// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE

#include <cmath>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <Eigen/Eigen>

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

#include "franka_test/VelCmd.h"
#include "print_pose.h"

namespace franka_test {

  template <typename T> int sgn(T val) {
      return (T(0) < val) - (val < T(0));
  }

    class CartesianVelInterface : public controller_interface::MultiInterfaceController<
                                            franka_hw::FrankaVelocityCartesianInterface>
    {
    private:

        franka_hw::FrankaVelocityCartesianInterface* velocity_cartesian_interface_;
        std::unique_ptr<franka_hw::FrankaCartesianVelocityHandle> velocity_cartesian_handle_;
        ros::Duration       elapsed_time_;

        ros::ServiceServer  service;
        // ros::Subscriber     vel_cmd_sub;
        ros::Subscriber     reset_sub;
        bool                read_message = false;
        // float               decay_rate   = 0.99;
        float               dt           = 0.2;
        float               max_force    = 500.0;

        std::array<double, 6> raw_cmd = {};
        std::array<double, 6> filtered_cmd = {};

        float alpha   = 0.995;
        double max_lin = 0.8/1000.;
        double max_ang = 2.5/1000.;

        ros::Subscriber     force_sub;
        bool                use_external_fts = false;
        std::array<double, 6> wrench {};

    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh)
        {
            service = nh.advertiseService("/vel_cmd", &CartesianVelInterface::cmd_callback, this);
            // vel_cmd_sub = nh.subscribe("/vel_cmd", 1, &CartesianVelInterface::cmd_callback, this);

            std::string arm_id;
            if (!nh.getParam("arm_id", arm_id)) {
                ROS_ERROR("CartesianVelInterface: Could not get parameter arm_id");
                return false;
            }

            if (!nh.getParam("alpha", alpha)) {
                ROS_ERROR("CartesianVelInterface could not parse filter parameter");
            }

            velocity_cartesian_interface_ = robot_hardware->get<franka_hw::FrankaVelocityCartesianInterface>();
            if (velocity_cartesian_interface_ == nullptr) {
                ROS_ERROR(
                  "CartesianVelInterface: Could not get Cartesian velocity interface "
                  "interface from hardware");
                return false;
            }



            try {
                velocity_cartesian_handle_ = std::make_unique<franka_hw::FrankaCartesianVelocityHandle>(
                                        velocity_cartesian_interface_->getHandle(arm_id + "_robot")
                                    );
            } catch (const hardware_interface::HardwareInterfaceException& e) {
                ROS_ERROR_STREAM(
                    "CartesianVelInterface: Exception getting Cartesian handle: " << e.what());
                return false;
            }

            reset_sub = nh.subscribe("/reset_control_commands", 1, &CartesianVelInterface::resetCallback, this);

            force_sub = nh.subscribe("/ee_wrench", 1, &CartesianVelInterface::forceCallback, this);

            return true;
        }

        void starting(const ros::Time&) {
            // current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
            elapsed_time_ = ros::Duration(0.0);
            ros::param::get("/max_force", max_force);
            ros::param::get("/dt", dt);
            ros::param::get("/use_external_fts", use_external_fts);
            raw_cmd = {{0., 0., 0., 0., 0., 0.}};
            filtered_cmd = {{0., 0., 0., 0., 0., 0.}};

        }

        void update(const ros::Time&, const ros::Duration& period) {

            // update the current pose
            // std::array<double, 16>  current_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;

            if (read_message) {
                // resetting the duration if there was a message
                elapsed_time_ = ros::Duration(0.);
                read_message = false; // reset the message
            }
            else {
                // updating if there wasn't, there could be something wrong
                elapsed_time_ += period;
                // TODO: add some decay if comms are lost
                // decay_rate
            }

            if (elapsed_time_.toSec() > dt*2.) {
              for (int i=0; i < 6; i++) {
                raw_cmd[i] = alpha * raw_cmd[i] + (1.0-alpha) * 0.;
              }
            }

            // check forces
            if (!use_external_fts){
              wrench = velocity_cartesian_handle_->getRobotState().O_F_ext_hat_K; // extimated external wrench actinv on stiffness frame relative to base frame
            }
            double norm_force = sqrt(pow(wrench[0],2)+pow(wrench[1],2)+pow(wrench[2],2));
            bool force_clamp = (norm_force > 0.75*max_force);
            // if (force_clamp) {
            //   ROS_INFO_STREAM("clipping cmd b/c norm of force > max_force");
            // }

            // get desired command
            std::array<double, 6> desired_cmd;
            for (int i=0; i < 6; i++) {
              if ((force_clamp)  && (i < 3) && (abs(raw_cmd[i]) > 1e-5)) {
                ROS_INFO_STREAM("clipping. i= " << i << ", wrench = " << wrench[i] << ", cmd = " << raw_cmd[i]);
              }
              if ((force_clamp) && (i < 3) && (signbit(wrench[i]) != signbit(raw_cmd[i]))) { // move away from force
                desired_cmd[i] = 0. - filtered_cmd[i];
              }
              else if ((force_clamp) && (i >= 3)) { // stop rotating
                desired_cmd[i] = 0. - filtered_cmd[i];
              }
              else {
                desired_cmd[i] = raw_cmd[i] - filtered_cmd[i];
              }
            }
            double lin_norm = sqrt(pow(desired_cmd[0],2)+pow(desired_cmd[1],2)+pow(desired_cmd[2],2));
            double ang_norm = sqrt(pow(desired_cmd[3],2)+pow(desired_cmd[4],2)+pow(desired_cmd[5],2));

            // apply to robot
            double clipped_cmd;
            for (int i=0; i < 6; i++) {
              clipped_cmd = desired_cmd[i];
              if ( (i < 3) && (lin_norm > max_lin) ){
                clipped_cmd = clipped_cmd/lin_norm*max_lin;
              }
              else if ( (i >=3) && (ang_norm > max_ang) ){
                clipped_cmd = clipped_cmd/ang_norm*max_ang;
              }
              filtered_cmd[i] = filtered_cmd[i] + clipped_cmd;
            }

            // OLD: alternate to wrench / max cmd checking above
            // for (int i=0; i < 6; i++) {
            //   filtered_cmd[i] = alpha * filtered_cmd[i] + (1.0-alpha) * raw_cmd[i];
            // }

            // FOR DEBUGGING
            // ROS_INFO_STREAM("  filtered_cmd  = x: " << filtered_cmd[0] << " raw_cmd  = x: " << raw_cmd[0]);
            // ROS_INFO_STREAM("  raw_cmd  = x: " << raw_cmd[0] << " y: " << raw_cmd[1] << " z: " << raw_cmd[2]);
            // ROS_INFO_STREAM("  filtered_cmd  = x: " << filtered_cmd[0] << " y: " << filtered_cmd[1] << " z: " << filtered_cmd[2]);
            // std::array<double, 6> vel2 = velocity_cartesian_handle_->getRobotState().O_dP_EE_c; // commanded ee vel
            // ROS_INFO_STREAM("  O_dP_EE_c  = x: " << vel2[0] << " y: " << vel2[1] << " z: " << vel2[2]);

            velocity_cartesian_handle_->setCommand(filtered_cmd);
        }

        bool cmd_callback(franka_test::VelCmd::Request &req,
                          franka_test::VelCmd::Response &res) {

            raw_cmd[0]=req.vel_cmd.linear.x;
            raw_cmd[1]=req.vel_cmd.linear.y;
            raw_cmd[2]=req.vel_cmd.linear.z;
            raw_cmd[3]=req.vel_cmd.angular.x;
            raw_cmd[4]=req.vel_cmd.angular.y;
            raw_cmd[5]=req.vel_cmd.angular.z;
            read_message = true;

            std::array<double, 16> pose = velocity_cartesian_handle_->getRobotState().O_T_EE; // _d means desired
            res.pose = print_pose(pose);
            std::array<double, 6> vel = velocity_cartesian_handle_->getRobotState().O_dP_EE_c; // _d means desired
            ROS_INFO("CartesianVel: ");
            ROS_INFO_STREAM("  xyz  = x: "  << vel[0] << " y: " << vel[1] << " z: " << vel[2]);
            ROS_INFO_STREAM("  rpy  = r: " << vel[3] << " p: " << vel[4] << " y: " << vel[5] );
            res.success = velocity_cartesian_handle_->getRobotState().control_command_success_rate > 0.5;
            if (!res.success){
              filtered_cmd = {{0., 0., 0., 0., 0., 0.}}; // robot stopped moving so need to ramp up again from scratch
            }
            return true;
        }

        void resetCallback(const std_msgs::Empty::ConstPtr& msg){
          raw_cmd = {{0., 0., 0., 0., 0., 0.}};
        }

      void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg){
        wrench = {{msg->wrench.force.x,msg->wrench.force.y,msg->wrench.force.z,
          msg->wrench.torque.x,msg->wrench.torque.y,msg->wrench.torque.z}};
      }

      };
}  // namespace franka_test

PLUGINLIB_EXPORT_CLASS(franka_test::CartesianVelInterface,
                       controller_interface::ControllerBase)
