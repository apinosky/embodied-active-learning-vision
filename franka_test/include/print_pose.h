
#pragma once


#include <cmath>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <ros/ros.h>

#include <geometry_msgs/Pose.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

namespace franka_test {

inline double constrainAngle(double x){
    x = fmod(x + M_PI,2*M_PI);
    if (x < 0)
        x += 2*M_PI;
    return x - M_PI;
}

inline geometry_msgs::Pose print_pose(const std::array<double, 16> &pose){
    tf2::Matrix3x3 m = tf2::Matrix3x3(pose[0],pose[4],pose[8],
                                      pose[1],pose[5],pose[9],
                                      pose[2],pose[6],pose[10]);
    tf2::Quaternion q;
    m.getRotation(q);
    double r,p,y;
    m.getRPY(r,p,y);

    // ROS_DEBUG("CartesianPose: " );
    // for (int i = 0; i<16 ; i++) {
    //     ROS_INFO_STREAM(pose[i]);
    // }
    ROS_INFO("CartesianPose: ");
    ROS_INFO_STREAM("  xyz  = x: "  << pose[12] << " y: " << pose[13] << " z: " << pose[14]);
    ROS_INFO_STREAM("  quat = x: " << q.getX() << " y: " << q.getY() << " z: " << q.getZ() << " w: " << q.getW());
    ROS_INFO_STREAM("  rpy  = r: " << r << " p: " << p << " y: " << y );

    geometry_msgs::Pose msg;
    msg.orientation = tf2::toMsg(q);
    msg.position.x = pose[12];
    msg.position.y = pose[13];
    msg.position.z = pose[14];

    return msg;
  }
}
