import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # Precomputation here
        
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value
        self.deterministic = False
        self.sigma = 0.1
        self.sigma_theta = 0.05

        ####################################


    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################

        def vector_to_T(x, y, theta):
            return np.array([
                [np.cos(theta), -np.sin(theta), x], 
                [np.sin(theta), np.cos(theta), y], 
                [0, 0, 1]
            ])
        
        def T_to_vector(T):
            theta = np.arctan2(T[1, 0], T[0, 0])
            x = T[0, 2]
            y = T[1, 2]
            return np.array([x, y, theta])
    
        # def T_odom():
        #     if not self.deterministic:
        #         odometry[0] = np.random.normal(odometry[0], self.sigma)
        #         odometry[1] = np.random.normal(odometry[1], self.sigma)
        #         odometry[2] = np.random.normal(odometry[2], self.sigma_theta)
        #     return vector_to_T(*odometry) 
        
        # updated_particles = np.array([
        #     T_to_vector(vector_to_T(*particle) @ T_odom()) for particle in particles
        # ])
        
        def sample_odometry(odometry, num_particles):
            if self.deterministic:
                return np.tile(odometry, (num_particles, 1))
            dx = np.random.normal(odometry[0], self.sigma, size=num_particles)
            dy = np.random.normal(odometry[1], self.sigma, size=num_particles)
            dtheta = np.random.normal(odometry[2], self.sigma_theta, size=num_particles)
            return np.column_stack((dx, dy, dtheta))
        
        sampled_odometry = sample_odometry(odometry, len(particles))

        updated_particles = np.array([
            T_to_vector(vector_to_T(*particle) @ vector_to_T(*odom_noise))
            for particle, odom_noise in zip(particles, sampled_odometry)
        ])
        
        return updated_particles
        ####################################
