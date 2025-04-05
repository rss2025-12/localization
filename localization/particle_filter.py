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
        
        # self.particle_lock = Lock()
        self.threshold = 1e-100
        self.particles_updated = False
        self.initial_pose_set = False
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
        
        self.initial_pose_set = True
        self.get_logger().info('Particles initialized.')


    def odom_callback(self, odom):
        if not self.initial_pose_set:
            return
        
        x = odom.twist.twist.linear.x
        y = odom.twist.twist.linear.y
        theta = odom.twist.twist.angular.z

        odometry = np.array([x, y, theta])

        # self.particles = self.motion_model.evaluate(self.particles, [0.1, 0, 0])
        self.particles = self.motion_model.evaluate(self.particles, odometry)
        self.update_pose()

        
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

        ### Updating particle weights
        # self.particle_lock.acquire()
        self.particle_weights = self.sensor_model.evaluate(self.particles, downsampled_scan)**(1/3)
        # self.particles_updated = True if self.particle_weights is not None else False
        self.particles_updated = True
        # self.particle_lock.release()
        self.particles, self.particle_weights = self.resample_all()
        self.update_pose()
 

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

        sin_sum = np.sum(np.sin(self.particles[:, 2]) * self.particle_weights)
        cos_sum = np.sum(np.cos(self.particles[:, 2]) * self.particle_weights)
        theta = np.arctan2(sin_sum, cos_sum)

        ### Convert orientation to quaternion ###
        quat = quaternion_from_euler(0, 0, theta)

        ### Publish Predicted Pose ###
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

    
    def resample_all(self):
        particles = self.particles.copy()
        particle_weights = self.particle_weights.copy()

        # particle_weights = np.exp(particle_weights) / np.sum(np.exp(particle_weights))
        particle_weights = particle_weights / np.sum(particle_weights)

        particle_indices = np.arange(len(particle_weights))
        choice_indices = np.random.choice(particle_indices, size=self.particle_count, p=particle_weights.reshape(-1), replace=True)
        resampled_particles = particles[choice_indices, :]
        dx = np.random.normal(resampled_particles[:,0], 0.1, size=self.particle_count).reshape(-1,1)
        dy = np.random.normal(resampled_particles[:,1], 0.1, size=self.particle_count).reshape(-1,1)
        dtheta = np.random.normal(resampled_particles[:,2], 0.1, size=self.particle_count).reshape(-1,1)
        resampled_particles = np.hstack((dx,dy,dtheta))

        # resampled_weights = np.ones(self.particle_count) / self.particle_count
        resampled_weights = particle_weights[choice_indices] / sum(particle_weights[choice_indices])
        self.get_logger().info(f'the particle weights are {resampled_weights}')
        
        return (resampled_particles, resampled_weights)
    

    def resample(self, ):
        particles = self.particles.copy()
        particle_weights = self.particle_weights.copy()

        particle_weights = np.exp(particle_weights) / np.sum(np.exp(particle_weights))
        # particle_weights = particle_weights / np.sum(particle_weights)
        particle_indices = np.arange(len(particle_weights))
        # choose particle indices to randomly sample
        choice_indices = np.random.choice(particle_indices, size=self.particle_count - len(particle_weights), p=particle_weights.reshape(-1))
        # get randomly selected particles and weights
        resampled_particles = particles[choice_indices, :]
        resampled_weights = particle_weights[choice_indices]
        # add new particles and weights to particles and particle_weights arrays
        new_particles = np.vstack((particles, resampled_particles))
        new_particle_weights = np.hstack((particle_weights, resampled_weights))
        new_particle_weights = new_particle_weights / np.sum(new_particle_weights) # normalize weights

        return (new_particles, new_particle_weights)
    
        # Delete particles with too low weight
        # if self.particles_updated:
        #     self.get_logger().info(f'inside update_pose conditional')

        #     ### Delete low-prob weights >>>
        #     # self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        #     # mask = self.particle_weights > self.threshold
        #     # # self.get_logger().info(f'the particles before are {self.particles}')
        #     # self.particles = self.particles[mask.reshape(-1)]
        #     # # self.get_logger().info(f'the particles after are {self.particles}')
        #     # self.get_logger().info(f'the particles weights before are {self.particle_weights}')
        #     # self.particle_weights = self.particle_weights[mask.reshape(-1)]
        #     # self.get_logger().info(f'the particles weights after are {self.particle_weights}')
        #     # self.get_logger().info(f'height: {len(self.particle_weights)}')
        #     # self.particle_weights = self.particle_weights / np.sum(self.particle_weights)
        #     ### <<< Delete low-prob weights

        #     self.particles, self.particle_weights = self.resample_all()

        #     # self.particle_weights /= np.linalg.norm(self.particle_weights)
            
        #     # x = array([[1,2],[2,3],[3,4]])
        #     # mask = [False,False,True]
        #     # x[~np.array(mask)]
        #     # # array([[1, 2],
        #     # #        [2, 3]])
        
        #     # resample particles
        #     # should we be resampling before or after deletion?

            
        #     # particle_indices = np.arange(len(self.particle_weights))
        #     # choice_indices = np.random.choice(particle_indices, size=self.particle_count - len(self.particle_weights), p=self.particle_weights.reshape(-1))
        #     # resampled_particles = self.particles[choice_indices, :]
        #     # resampled_weights = self.particle_weights[choice_indices]
        #     # self.particles = np.vstack((self.particles, resampled_particles))
        #     # self.particle_weights = np.hstack((self.particle_weights, resampled_weights))
        #     # self.particle_weights = self.particle_weights / np.sum(self.particle_weights)

        # self.get_logger().info(f'The particles weights post update are {self.particle_weights}')


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
