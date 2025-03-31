import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # Precomputation here
        
        node.declare_parameter('deterministic', False)
        node.declare_parameter('noise_std', 1.0)

        self.noise_std = node.get_parameter('noise_std').get_parameter_value().double_value
        self.deterministic = node.get_parameter('deterministic').get_parameter_value().bool_value

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

        def T_odom():
            if self.deterministic:
                return vector_to_T(*odometry)
            return vector_to_T(*np.random.normal(odometry, self.noise_std)) 

        updated_particles = np.array([
            T_to_vector(vector_to_T(*particle) @ T_odom()) for particle in particles
        ])
        
        return updated_particles

        ####################################
