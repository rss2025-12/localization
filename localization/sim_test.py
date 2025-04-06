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
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        # Timer to send commands at regular intervals
        self.time_increment = 0.25
        self.timer = self.create_timer(self.time_increment, self.timer_callback)  # Call every 1 second
        self.elapsed_time = 0  # Track elapsed time

        self.steering_schedule = {
            0.0: 0.0,
            21.5: -0.205,
            22.5: 0.0,
            35.25: -0.198,
            36.25: 0.0,
            48.75: -0.202,
            49.75: 0.0,
            51.75: 0.203,
            52.75: 0.0,
            56.5: -0.1,
            59.5: 0.0,
            66.0: 0.09,
            67.0: 0.0,
            69.0: 0.205,
            70.0: 0.0
        }

        self.speed = 0.0
        self.steering_angle = 0.0

        # Set initial speed, angle, pose at the start
        drive_msg = AckermannDriveStamped()
        drive_msg.header.stamp = self.get_clock().now().to_msg()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = self.steering_angle
        self.drive_pub.publish(drive_msg)
        self.send_initial_pose()

    def send_initial_pose(self):
        # Set the initial pose in the 'map' frame
        x = 0.0
        y = -1.0
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
        for _ in range(10):
            self.initial_pose_pub.publish(initial_pose_msg)
            self.pose_pub.publish(pose_msg)
        self.get_logger().info(f'Initial pose {x, y, theta} sent')

    def timer_callback(self):
        # Set speed
        if self.elapsed_time == 0:
            self.speed = 2.5

        # Steering schedule
        if self.elapsed_time in self.steering_schedule:
            self.steering_angle = self.steering_schedule[self.elapsed_time]

        # Stop
        if self.elapsed_time == 71:
            self.speed = 0.0
            self.get_logger().info("Test finished")

        self.elapsed_time += self.time_increment

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
