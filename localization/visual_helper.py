import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, TransformStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

import numpy as np
import os, csv

class VisualHelper(Node):
    def __init__(self):
        super().__init__('visual_helper')

        # Estimated pose
        self.estimated_pose_sub = self.create_subscription(
            Odometry,
            '/pf/pose/odom',
            self.pose_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan_replay',
            self.scan_callback,
            10
        )

        self.tf_broadcaster = TransformBroadcaster(self)
        self.scan_pub = self.create_publisher(LaserScan, '/scan_overlay', 1)
     
        self.estimated_pose = None

    def pose_callback(self, odom_msg):
        x = odom_msg.pose.pose.position.x
        y = odom_msg.pose.pose.position.y
        quat = odom_msg.pose.pose.orientation

        self.estimated_pose = (
            x,
            y,
            (quat.x, quat.y, quat.z, quat.w)
        )

    def scan_callback(self, scan_msg):
        if self.estimated_pose is None:
            return

        x, y, quat = self.estimated_pose

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'scan_overlay'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        self.tf_broadcaster.sendTransform(t)

        new_scan = scan_msg
        new_scan.header.stamp = self.get_clock().now().to_msg()
        new_scan.header.frame_id = 'scan_overlay'

        num_points = 100
        total_points = len(scan_msg.ranges)
        step_size = max(1, total_points // num_points)

        downsampled_ranges = []
        downsampled_angles = []
        for i in range(0, total_points, step_size):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            downsampled_ranges.append(scan_msg.ranges[i])
            downsampled_angles.append(angle)

        new_scan.ranges = downsampled_ranges
        new_scan.angle_min = downsampled_angles[0] if downsampled_angles else scan_msg.angle_min
        new_scan.angle_max = downsampled_angles[-1] if downsampled_angles else scan_msg.angle_max
        new_scan.angle_increment = (downsampled_angles[-1] - downsampled_angles[0]) / len(downsampled_angles) if len(downsampled_angles) > 1 else scan_msg.angle_increment

        self.scan_pub.publish(new_scan)

def main(args=None):
    rclpy.init(args=args)
    node = VisualHelper()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
