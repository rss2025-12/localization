from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel
from threading import Lock

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose, Point, Quaternion
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler

from rclpy.node import Node
import rclpy
import numpy as np

assert rclpy

class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('deterministic', False)
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
        self.particle_pub= self.create_publisher(PoseArray, "/particles", 1)

        # self.particles = np.zeros((self.particle_count, 3))
        # self.particle_weights = np.ones(self.particle_count)

        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)
        
        self.particle_lock = Lock()
        # self.particle_lock.acquire()
        self.threshold = 1e-100
        self.particles_updated = False
        self.pose_set = False
        self.get_logger().info("=============+READY+=============")
    
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
        self.get_logger().info('Initial pose recieved.')
        
        x = pose.pose.pose.position.x
        y = pose.pose.pose.position.y

        qx = pose.pose.pose.orientation.x
        qy = pose.pose.pose.orientation.y
        qz = pose.pose.pose.orientation.z
        qw = pose.pose.pose.orientation.w
        
        _, _, theta = euler_from_quaternion([qx, qy, qz, qw])

        sigma = 0.5
        sigma_theta = 0.1

        self.particles = np.column_stack([
            np.random.normal(x, sigma, self.particle_count),
            np.random.normal(y, sigma, self.particle_count),
            np.random.normal(theta, sigma_theta, self.particle_count)
        ])

        self.particle_weights = np.full(self.particle_count, 1 / self.particle_count)
        
        # self.get_logger().info('Particles initialized.')
        self.pose_set = True
        
        
    def laser_callback(self, laser):
        """
        Uses scan data to generate weights for the particles.
        Returns nothing.
        """
        if not self.pose_set:
            return
        # self.get_logger().info(f'Laser callback')

        desired_sample_num = self.sensor_model.num_beams_per_particle
        min_angle = -self.sensor_model.scan_field_of_view/2
        max_angle = self.sensor_model.scan_field_of_view/2
        angle_vector = np.arange(laser.angle_min, laser.angle_max, laser.angle_increment)

        downsampled_scan = np.interp(
            np.linspace(min_angle, max_angle, desired_sample_num, endpoint = True), 
            angle_vector, 
            laser.ranges
        )

        self.particle_lock.acquire()
        self.particle_weights = self.sensor_model.evaluate(self.particles, downsampled_scan)
        # new_weights = self.sensor_model.evaluate(self.particles, downsampled_scan)
        # self.particle_weights = new_weights if new_weights is not None else self.particle_weights
        self.particle_lock.release()
        self.particles_updated = True if self.particle_weights is not None else False
        # self.get_logger().info(f'Map updated: {self.sensor_model.map_set}')
        # self.get_logger().info(f'The particles weights pre-update are {self.particle_weights}')
        self.update_pose()


    def odom_callback(self, odom):
        if not self.pose_set:
            return
        
        self.particle_lock.acquire()
        self.get_logger().info('Odom callback')

        # x = odom.pose.pose.position.x
        # y = odom.pose.pose.position.y

        x = odom.twist.twist.linear.x
        y = odom.twist.twist.linear.y

        # qx = odom.pose.pose.orientation.x
        # qy = odom.pose.pose.orientation.y
        # qz = odom.pose.pose.orientation.z
        # qw = odom.pose.pose.orientation.w

        # wx = twist.twist.angular.x
        # wy = twist.twist.angular.y
        theta = odom.twist.twist.angular.z

        # _, _, theta = euler_from_quaternion([qx, qy, qz, qw])

        odometry = np.array([x, y, theta])
        self.get_logger().info(f'odom is {[x, y, theta]}')


        self.particles = self.motion_model.evaluate(self.particles, odometry)
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
        # self.get_logger().info(f'The particles weights pre-update are {self.particle_weights}')

        ### Take mean and circular mean of measurements: ###
        # particles = self.particles * self.particle_weights[:,np.newaxis]
        # x, y = np.mean(particles[:, :2], axis=0)
        # thetas = particles[:,2]
        # theta = np.arctan2((np.sum(np.sin(thetas))), (np.sum(np.cos(thetas))))

        # x = np.sum(self.particles[:, 0] * self.particle_weights)
        # y = np.sum(self.particles[:, 1] * self.particle_weights)

        # theta = np.arctan2(
        #     np.sum(np.sin(self.particles[:, 2]) * self.particle_weights),
        #     np.sum(np.cos(self.particles[:, 2]) * self.particle_weights)
        # )
        # ###
        
        # # particle = np.hstack((np.mean(particles[:, :2], axis=0), theta_mean))

        # ### Publishing pose estimate ###
        # odom_msg = Odometry()
        # odom_msg.header.stamp = self.get_clock().now().to_msg()
        # odom_msg.header.frame_id = "map"
        # odom_msg.child_frame_id = self.particle_filter_frame

        # odom_msg.pose.pose.position.x = x
        # odom_msg.pose.pose.position.y = y
        # odom_msg.pose.pose.position.z = 0.0

        # quat = quaternion_from_euler(0, 0, theta)
        # odom_msg.pose.pose.orientation.x = quat[0]
        # odom_msg.pose.pose.orientation.y = quat[1]
        # odom_msg.pose.pose.orientation.z = quat[2]
        # odom_msg.pose.pose.orientation.w = quat[3]

        # odom_msg.pose.pose.orientation = Quaternion(
        #     x = quat[0],
        #     y = quat[1],
        #     z = quat[2],
        #     w = quat[3],
        # )

        # self.odom_pub.publish(odom_msg)
        ###

        ### Publishing particles ###
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
        # self.get_logger().info(f'Particle poses are {self.particles}')
        ###

        # Delete particles with too low weight
        if self.particles_updated:
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
            mask = self.particle_weights > self.threshold
            # self.get_logger().info(f'the particles before are {self.particles}')
            self.particles = self.particles[mask.reshape(-1)]
            # self.get_logger().info(f'the particles after are {self.particles}')
            self.get_logger().info(f'the particles weights before are {self.particle_weights}')
            self.particle_weights = self.particle_weights[mask.reshape(-1)]
            self.get_logger().info(f'the particles weights after are {self.particle_weights}')
            self.get_logger().info(f'height: {len(self.particle_weights)}')
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

            # self.particle_weights /= np.linalg.norm(self.particle_weights)
            
            # x = array([[1,2],[2,3],[3,4]])
            # mask = [False,False,True]
            # x[~np.array(mask)]
            # # array([[1, 2],
            # #        [2, 3]])
        
            # resample particles
            # should we be resampling before or after deletion?
            particle_indices = np.arange(len(self.particle_weights))
            choice_indices = np.random.choice(particle_indices, size=self.particle_count - len(self.particle_weights), p=self.particle_weights.reshape(-1))
            resampled_particles = self.particles[choice_indices, :]
            resampled_weights = self.particle_weights[choice_indices]
            self.particles = np.vstack((self.particles, resampled_particles))
            self.particle_weights = np.hstack((self.particle_weights, resampled_weights))
            self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

        self.get_logger().info(f'The particles weights post update are {self.particle_weights}')
        return

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
