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

        self.scan_pub.publish(new_scan)

def main(args=None):
    rclpy.init(args=args)
    node = VisualHelper()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
