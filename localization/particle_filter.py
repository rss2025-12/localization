from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from rclpy.node import Node
import rclpy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion, PoseStamped, TransformStamped
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster
assert rclpy

import numpy as np
from threading import Lock

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('deterministic', False)
        self.declare_parameter('odom_invert', False)
        self.declare_parameter('sensor_model_frequency', 10.0)
        self.declare_parameter('particle_filter_frame', "/base_link")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")
        self.declare_parameter('particle_count', 200)

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value


        self.odom_invert = self.get_parameter("odom_invert").get_parameter_value().bool_value
        self.sensor_freq = self.get_parameter("sensor_model_frequency").get_parameter_value().double_value

        self.particle_count = self.get_parameter("particle_count").get_parameter_value().integer_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.particle_pub = self.create_publisher(PoseArray, "/debug_particles", 1)
        self.visualize_particles = False

        # Slime path
        self.path_pub = self.create_publisher(Path, "/odom_path", 1)
        self.path = Path()
        self.path.header.stamp = self.get_clock().now().to_msg()
        self.path.header.frame_id = "map"
        self.visualize_path = False

        # Visualize lidar data
        self.tf_broadcaster = TransformBroadcaster(self)
        self.visualize_laser = False

        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)
        self.is_free = self.sensor_model.is_free

        self.last_odom_time = None
        self.last_laser_time = None
        self.last_path_time = None
        self.initial_pose_set = False
        self.get_logger().info("=============+READY+=============")

        self.particle_lock = Lock()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.


    def pose_callback(self, pose):
        """
        Given a initial pose, generates some particles around the pose
        """
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y

        qx = pose.pose.pose.orientation.x
        qy = pose.pose.pose.orientation.y
        qz = pose.pose.pose.orientation.z
        qw = pose.pose.pose.orientation.w
        _, _, theta = euler_from_quaternion([qx, qy, qz, qw])

        self.get_logger().info(f'Initial pose {x, y, theta} recieved')

        sigma = 0.5
        sigma_theta = 0.1

        self.particles = np.column_stack([
            np.random.normal(x, sigma, self.particle_count),
            np.random.normal(y, sigma, self.particle_count),
            np.random.normal(theta, sigma_theta, self.particle_count)
        ])
        self.particle_weights = np.full(self.particle_count, 1 / self.particle_count)

        self.initial_pose_set = True
        self.get_logger().info('Particles initialized')


    def odom_callback(self, odom):
        if not self.initial_pose_set:
            return

        current_timestamp = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9

        if self.last_odom_time is None:
            self.last_odom_time = current_timestamp
            current_timestamp += 1/50 # Average frequency of /odom

        dt = current_timestamp - self.last_odom_time
        self.last_odom_time = current_timestamp

        x = odom.twist.twist.linear.x
        y = odom.twist.twist.linear.y
        theta = odom.twist.twist.angular.z

        odometry = np.array([x * dt, y * dt, theta * dt]) if not self.odom_invert else -np.array([x * dt, y * dt, theta * dt])

        self.particle_lock.acquire()
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        self.update_pose()
        self.particle_lock.release()


    def laser_callback(self, laser):
        """
        Uses scan data to generate weights for the particles.
        Returns nothing.
        """
        if not self.initial_pose_set:
            return

        ### Downsizing laserscan to fit sensor_model scan ###
        desired_sample_num = self.sensor_model.num_beams_per_particle
        min_angle = -self.sensor_model.scan_field_of_view/2
        max_angle = self.sensor_model.scan_field_of_view/2
        angle_vector = np.arange(laser.angle_min, laser.angle_max, laser.angle_increment)

        downsampled_scan = np.interp(
            np.linspace(min_angle, max_angle, desired_sample_num, endpoint = True),
            angle_vector,
            laser.ranges
        )

        ### Updating particle weights ###
        self.particle_lock.acquire()
        self.particle_weights = self.sensor_model.evaluate(self.particles, downsampled_scan)**(1/3)
        self.particles, self.particle_weights = self.resample()
        self.update_pose()
        self.particle_lock.release()



    def update_pose(self):
        """
        Update the pose of the rover by publishing the "average" particle transform

        args:
            self

        returns:
            None
        """
        ### Take mean and circular mean of measurements ###
        x = np.sum(self.particles[:, 0] * self.particle_weights)
        y = np.sum(self.particles[:, 1] * self.particle_weights)

        # if(np.max(np.abs([x, y])) < 10**-10):
        #     # self.get_logger().info(f'x, y flicker to 0 blocked: {x, y}')
        #     return

        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.particle_weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.particle_weights)
        theta = np.arctan2(sin_sum, cos_sum)
        quat = quaternion_from_euler(0, 0, theta)

        ### Publish predicted pose ###
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        odom_msg.child_frame_id = self.particle_filter_frame

        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = Quaternion(
            x = quat[0],
            y = quat[1],
            z = quat[2],
            w = quat[3],
        )

        self.odom_pub.publish(odom_msg)

        if self.visualize_laser is True:
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'map'
            t.child_frame_id = 'laser_model'
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = 0.0
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            self.tf_broadcaster.sendTransform(t)

        ### Publishing path ###
        if self.visualize_path is True:
            current_time = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec*10**-9
            if(self.last_path_time is None or current_time > self.last_path_time + 1/3):
                self.last_path_time = current_time
                pose_stamped = PoseStamped()
                pose_stamped.header = odom_msg.header
                pose_stamped.pose = odom_msg.pose.pose
                self.path.poses.append(pose_stamped)

                self.path_pub.publish(self.path)

        ### Publishing particles ###
        if self.visualize_particles is True:
            visual_particles = PoseArray()
            visual_particles.header.stamp = self.get_clock().now().to_msg()
            visual_particles.header.frame_id = "map"

            visual_particles.poses = []
            for particle in self.particles:
                q = quaternion_from_euler(0, 0, particle[2])
                quaternion_msg = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

                pose = Pose(
                    position=Point(x=particle[0], y=particle[1], z=0.0),
                    orientation=quaternion_msg
                )
                visual_particles.poses.append(pose)

            self.particle_pub.publish(visual_particles)

        self.updating = False


    def resample(self):
        ### Copying and normalizing particles ###
        particles = self.particles.copy()
        particle_weights = self.particle_weights.copy()
        particle_weights = particle_weights / np.sum(particle_weights)

        ### Resample only if Neff is low ###

        neff = 1.0 / np.sum(particle_weights**2)
        if neff > 0.6 * self.particle_count: # Between 0.5 - 0.7
            return particles, particle_weights

        ### Resampling ###
        particle_indices = np.arange(len(particle_weights))
        choice_indices = np.random.choice(particle_indices, size=self.particle_count, p=particle_weights.reshape(-1), replace=True)
        resampled_particles = particles[choice_indices, :]
        resampled_weights = particle_weights[choice_indices]

        for i in range(len(resampled_particles)):
            p = particles[i]
            x,y = p[:2]
            if self.is_free(x,y) == 0:
                resampled_weights[i] = 0

        resampled_weights = resampled_weights / sum(resampled_weights)
        return (resampled_particles, resampled_weights)


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
