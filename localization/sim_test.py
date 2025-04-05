import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped, TransformStamped
import tf_transformations

import numpy as np

class SimTest(Node):
    def __init__(self):
        super().__init__('sim_test')
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.pose_pub = self.create_publisher(Pose, '/pose', 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initial_pose', 10)

        self.speed = 0.0
        self.steering_angle = 0.0

        # Timer to send commands at regular intervals
        self.timer = self.create_timer(1.0, self.timer_callback)  # Call every 1 second
        self.elapsed_time = 0  # Track elapsed time

        # Send initial pose at the start
        self.send_initial_pose()

    def send_initial_pose(self):
        # Set the initial pose at (0, 0, pi) in the 'map' frame
        x = 0.0
        y = 0.0
        theta = np.pi

        # Set robot pose
        pose_msg = Pose()
        pose_msg.position.x = x
        pose_msg.position.y = y

        quaternion = tf_transformations.quaternion_from_euler(0, 0, theta)
        pose_msg.orientation.y = quaternion[1]
        pose_msg.orientation.z = quaternion[2]
        pose_msg.orientation.w = quaternion[3]

        # Set initial pose
        initial_pose_msg = PoseWithCovarianceStamped()
        initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
        initial_pose_msg.header.frame_id = 'map'

        initial_pose_msg.pose.pose.position.x = x
        initial_pose_msg.pose.pose.position.y = y
        initial_pose_msg.pose.pose.position.z = 0.0

        # Orientation (pi radians, facing backwards along the x-axis)
        # Quaternion for a 180-degree rotation (π radians) around the Z-axis
        initial_pose_msg.pose.pose.orientation.x = 0.0
        initial_pose_msg.pose.pose.orientation.y = 0.0
        initial_pose_msg.pose.pose.orientation.z = np.sin(theta / 2)
        initial_pose_msg.pose.pose.orientation.w = np.cos(theta / 2)

        # Publish the initial pose
        self.pose_pub.publish(pose_msg)
        self.initial_pose_pub.publish(initial_pose_msg)
        self.get_logger().info(f'Initial pose set to {x, y, theta}')

    def timer_callback(self):
        # Example logic for sending drive commands based on elapsed time
        if self.elapsed_time == 0:  # 5 seconds: Forward
            self.speed = 2.5
            self.steering_angle = 0.0
            self.get_logger().info(f"Time {self.elapsed_time}s")
        elif self.elapsed_time == 20: # 10 seconds: Left turn
            self.speed = 2.5
            self.steering_angle = 0.1
            self.get_logger().info(f"Time {self.elapsed_time}s")
        elif self.elapsed_time == 21:  # 15 seconds: Forward
            self.speed = 2.5
            self.steering_angle = -0.1
            self.get_logger().info(f"Time {self.elapsed_time}s")
        elif self.elapsed_time == 22: # 20 seconds: Left Turn
            self.speed = 2.5
            self.steering_angle = 0
            self.get_logger().info(f"Time {self.elapsed_time}s")
        self.elapsed_time += 1

        # Create and publish the drive command message
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = self.steering_angle
        self.drive_pub.publish(drive_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SimTest()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
