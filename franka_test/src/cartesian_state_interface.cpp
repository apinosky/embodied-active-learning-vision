  // adapted from https://github.com/frankaemika/franka_ros/blob/melodic-devel/franka_example_controllers/src/cartesian_pose_example_controller.cpp
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
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_model_interface.h>

#include <hardware_interface/hardware_interface.h>
#include <hardware_interface/robot_hw.h>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <ros/node_handle.h>
#include <ros/time.h>

#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>

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

        ros::Publisher      ee_pose_pub;
        ros::Publisher      ee_vel_pub;
        ros::Timer          ee_timer;

        ros::Subscriber     force_sub;
        ros::Publisher      force_pub;
        bool                use_external_fts = false;
        std::array<double, 6> filtered_wrench =  {{0., 0., 0., 0., 0., 0.}};
        float               alpha   = 0.95;
        float               ft_yaw_offset = 0;
        float               sensor_mass = 0.45;

    public:
        bool init(hardware_interface::RobotHW* robot_hardware, ros::NodeHandle& nh)
        {
            ros::param::get("/use_external_fts", use_external_fts);
            ros::param::get("/ft_yaw_offset", ft_yaw_offset);
            ros::param::get("/sensor_mass", sensor_mass);

            ee_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/ee_pose", 100);
            ee_vel_pub = nh.advertise<geometry_msgs::TwistStamped>("/ee_vel", 100);

            std::string arm_id;
            if (!nh.getParam("arm_id", arm_id)) {
                ROS_ERROR("CartesianStateController: Could not get parameter arm_id");
                return false;
            }


            state_interface_ = robot_hardware->get<franka_hw::FrankaStateInterface>();
            try {
                state_handle_ =  std::make_unique<franka_hw::FrankaStateHandle>(
                                        state_interface_->getHandle(arm_id + "_robot")
                                    );

            } catch (const hardware_interface::HardwareInterfaceException& e) {
                ROS_ERROR_STREAM(
                    "CartesianStateController: Exception getting state handle: " << e.what());
                return false;
            }

            model_interface_ = robot_hardware->get<franka_hw::FrankaModelInterface>();
            if (model_interface_ == nullptr) {
                ROS_ERROR(
                  "CartesianStateController: Error getting model interface from hardware "
                  "interface from hardware");
                return false;
            }

            try {
                model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
                                        model_interface_->getHandle(arm_id + "_model")
                                    );
            } catch (const hardware_interface::HardwareInterfaceException& e) {
                ROS_ERROR_STREAM(
                    "CartesianStateController: Exception getting Model handle: " << e.what());
                return false;
            }
            // timer for publishing pose
            ee_timer = nh.createTimer(ros::Duration(0.01), &CartesianStateInterface::timerCallback,this);

            force_pub = nh.advertise<geometry_msgs::WrenchStamped>("/ee_wrench", 100);
            force_sub = nh.subscribe("/netft_data", 1, &CartesianStateInterface::forceCallback, this);

            return true;
        }

        void starting(const ros::Time&) {
            // placeholder
            ros::param::get("/use_external_fts", use_external_fts);
            ros::param::get("/ft_yaw_offset", ft_yaw_offset);
            ros::param::get("/sensor_mass", sensor_mass);
        }

        void update(const ros::Time&, const ros::Duration& period) {
            // placeholder
        }

        void timerCallback(const ros::TimerEvent& event) {
          geometry_msgs::PoseStamped stamped_msg;
          std::array<double, 16> pose = state_handle_->getRobotState().O_T_EE; // _d means desired
          stamped_msg.header.stamp = ros::Time::now();

          tf2::Matrix3x3 m = tf2::Matrix3x3(pose[0],pose[4],pose[8],
                                            pose[1],pose[5],pose[9],
                                            pose[2],pose[6],pose[10]);
          tf2::Quaternion q;
          m.getRotation(q);
          geometry_msgs::Pose msg;
          msg.orientation = tf2::toMsg(q);
          msg.position.x = pose[12];
          msg.position.y = pose[13];
          msg.position.z = pose[14];

          // stamped_msg.header = img_msg->header;
          stamped_msg.header.frame_id = "panda_joint8";
          stamped_msg.pose = msg;
          ee_pose_pub.publish(stamped_msg);

          geometry_msgs::TwistStamped stamped_vel_msg;
          stamped_vel_msg.header.stamp = ros::Time::now();
          geometry_msgs::Twist vel_msg;
          // std::array<double, 7> joint_vel = state_handle_->getRobotState().dq; // measured joint velocities
          // // get jacobian
          // std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
          // // convert to eigen
          // Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
          // Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(joint_vel.data());
          // Eigen::Matrix<double, 6, 1> vel = jacobian*dq;
          std::array<double, 6> vel = state_handle_->getRobotState().O_dP_EE_c; // _d means desired
          vel_msg.linear.x  = vel[0];
          vel_msg.linear.y  = vel[1];
          vel_msg.linear.z  = vel[2];
          vel_msg.angular.x = vel[3];
          vel_msg.angular.y = vel[4];
          vel_msg.angular.z = vel[5];
          stamped_vel_msg.header.frame_id = "panda_joint8";
          stamped_vel_msg.twist = vel_msg;
          ee_vel_pub.publish(stamped_vel_msg);

          // ROS_INFO_STREAM("  jacobian  = x: " << vel[0] << " y: " << vel[1] << " z: " << vel[2]);
          // std::array<double, 6> vel2 = state_handle_->getRobotState().O_dP_EE_c; // commanded ee vel
          // ROS_INFO_STREAM("  O_dP_EE_c  = x: " << vel2[0] << " y: " << vel2[1] << " z: " << vel2[2]);

          if (!use_external_fts){
            std::array<double, 6> wrench = state_handle_->getRobotState().O_F_ext_hat_K; // extimated external wrench acting on stiffness frame relative to base frame

            for (int i=0; i < 6; i++) {
              filtered_wrench[i] = alpha * filtered_wrench[i] + (1.0-alpha) * wrench[i];
            }


            geometry_msgs::WrenchStamped stamped_wrench_msg;
            stamped_wrench_msg.header = stamped_vel_msg.header;
            stamped_wrench_msg.header.frame_id = "ee_frame";
            stamped_wrench_msg.wrench.force.x = filtered_wrench[0];
            stamped_wrench_msg.wrench.force.y = filtered_wrench[1];
            stamped_wrench_msg.wrench.force.z = filtered_wrench[2];
            stamped_wrench_msg.wrench.torque.x = filtered_wrench[3];
            stamped_wrench_msg.wrench.torque.y = filtered_wrench[4];
            stamped_wrench_msg.wrench.torque.z = filtered_wrench[5];
            force_pub.publish(stamped_wrench_msg);
          }
        }

        void forceCallback(const geometry_msgs::WrenchStamped::ConstPtr& msg){
          if (use_external_fts){
            std::array<double, 6> tmp_wrench = {{msg->wrench.force.x,msg->wrench.force.y,msg->wrench.force.z,
              msg->wrench.torque.x,msg->wrench.torque.y,msg->wrench.torque.z}};

            std::array<double, 16> pose = state_handle_->getRobotState().O_T_EE; // _d means desired

            // added for visualization purposes
            static tf::TransformBroadcaster br;
            tf::Transform transform;
            transform.setOrigin( tf::Vector3(pose[12], pose[13], pose[14]));
            tf::Quaternion q;
            q.setRPY(0, 0, 0);
            transform.setRotation(q);
            br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "ee_frame"));

            // rotate ft as needed
            q.setRPY(0, 0, ft_yaw_offset);
            Eigen::Quaterniond ft_quat;
            tf::quaternionTFToEigen(q,ft_quat);
            Eigen::Matrix<double, 3, 3> ft_mat = ft_quat.toRotationMatrix();

            // convert to eigen
            Eigen::Map<Eigen::Matrix<double, 6,1>> mat_wrench(tmp_wrench.data());
            Eigen::Map<Eigen::Matrix<double, 4, 4>> mat_trans(pose.data());
            Eigen::Matrix<double, 3, 3> mat_rot = mat_trans.topLeftCorner(3,3);
            mat_rot = mat_rot*ft_mat;

            // apply gravity offset
            Eigen::Matrix<double, 3,1> gravity_force;
            gravity_force << 0,0,9.81*sensor_mass;
            gravity_force = mat_rot.transpose()*gravity_force + gravity_force;
            mat_wrench.head(3) += gravity_force;

            // get adjoint
            Eigen::Matrix<double, 6, 6> adj;
            adj.topLeftCorner(3, 3)     =  mat_rot.transpose();
            adj.topRightCorner(3, 3).setZero();
            adj.bottomRightCorner(3, 3)  = mat_rot.transpose();
            adj.bottomLeftCorner(3, 3).setZero();

            Eigen::Matrix<double, 6,1> wrench_base = adj*mat_wrench;

            for (int i=0; i < 6; i++) {
              filtered_wrench[i] = alpha * filtered_wrench[i] + (1.0-alpha) * wrench_base[i];
            }

            geometry_msgs::WrenchStamped new_msg;
            new_msg.header = msg->header;
            new_msg.header.frame_id = "ee_frame";
            new_msg.wrench.force.x = filtered_wrench[0];
            new_msg.wrench.force.y = filtered_wrench[1];
            new_msg.wrench.force.z = filtered_wrench[2];
            new_msg.wrench.torque.x = filtered_wrench[3];
            new_msg.wrench.torque.y = filtered_wrench[4];
            new_msg.wrench.torque.z = filtered_wrench[5];
            force_pub.publish(new_msg);
          }
        }
    };
}  // namespace franka_test

PLUGINLIB_EXPORT_CLASS(franka_test::CartesianStateInterface,
                       controller_interface::ControllerBase)
