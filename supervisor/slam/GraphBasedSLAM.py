
# class EKFSlam(Slam):
#
#     def __init__(self, supervisor_interface, slam_cfg, step_time):
#         """
#         Initializes an object of the EKFSlam class
#         :param supervisor_interface: The interface to interact with the robot supervisor
#         :param slam_cfg: The configuration for the SLAM algorithm
#         :param step_time: The discrete time that a single simulation cycle increments
#         """
#         # Bind the supervisor interface
#         self.supervisor = supervisor_interface
#         # Extract relevant configurations
#         self.dt = step_time
#         self.distance_threshold = slam_cfg["ekf_slam"]["distance_threshold"]
#         self.robot_state_size = slam_cfg["robot_state_size"]
#         self.landmark_state_size = slam_cfg["landmark_state_size"]
#         self.sensor_noise = np.diag([slam_cfg["sensor_noise"]["detected_distance"],
#                                      np.deg2rad(slam_cfg["sensor_noise"]["detected_angle"])]) ** 2
#         self.motion_noise = np.diag([slam_cfg["ekf_slam"]["motion_noise"]["x"],
#                                      slam_cfg["ekf_slam"]["motion_noise"]["y"],
#                                      np.deg2rad(slam_cfg["ekf_slam"]["motion_noise"]["theta"])]) ** 2
#         # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
#         self.mu = np.zeros((self.robot_state_size, 1))
#         # The state covariance, initially set to absolute certainty of the initial robot pose
#         self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size))
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle
import numpy as np
from math import *
from models.Pose import Pose

class GraphBasedSLAM(Slam):

    def __init__(self, supervisor_interface, slam_cfg, step_time):
        """
        Initializes an object of the GraphBasedSLAM class
        :param supervisor_interface: The interface to interact with the robot supervisor
        :param slam_cfg: The configuration for the SLAM algorithm
        :param step_time: The discrete time that a single simulation cycle increments
        """
        # Bind the supervisor interface
        self.supervisor = supervisor_interface

        # Extract relevant configurations
        self.dt = step_time
        self.distance_threshold = slam_cfg["graph_based_slam"]["distance_threshold"]
        self.robot_state_size = slam_cfg["robot_state_size"]
        self.landmark_state_size = slam_cfg["landmark_state_size"]
        self.sensor_noise = np.diag([slam_cfg["sensor_noise"]["detected_distance"],
                                     np.deg2rad(slam_cfg["sensor_noise"]["detected_angle"])]) ** 2
        self.motion_noise = np.diag([slam_cfg["ekf_slam"]["motion_noise"]["x"],
                                     slam_cfg["ekf_slam"]["motion_noise"]["y"],
                                     np.deg2rad(slam_cfg["ekf_slam"]["motion_noise"]["theta"])]) ** 2
        # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
        self.mu = np.zeros((self.robot_state_size, 1))
        # The state covariance, initially set to absolute certainty of the initial robot pose
        self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size))


        self.step_counter = 0

    def get_estimated_pose(self):
        """
        Returns the estimated pose of the robot
        """
        return Pose(self.mu[0, 0], self.mu[1, 0], self.mu[2, 0])

    def get_landmarks(self):
        """
        Returns the estimated landmark positions
        """
        lm = [(1,2), (3,3), (1,3)]
        return [(x, y) for (x, y) in lm]

    def update(self, u, z):
        """
        Executes an update cycle of the SLAM algorithm
        :param u: motion command
        :param z: List of measurements
        """
        self.mu = self.motion_model(self.mu, u, self.dt)
        J = self.jaco_motion_model(self.mu, u, self.dt)  # calculate jacobian matrix
        self.Sigma = J.T @ self.Sigma @ J + self.motion_noise  # update covariance matrix
        self.step_counter += 1



    @staticmethod
    def jaco_motion_model(x, u, dt):
        v, w = u[0, 0], u[1, 0]
        s1, s12 = sin(x[2, 0]),  sin(x[2, 0] + dt*w)
        c1, c12 = cos(x[2, 0]),  cos(x[2, 0] + dt*w)
        # No angular velocity means following a straight line
        if w == 0:
            G = np.array([[1, 0, -dt*s1*v],
                          [0, 1,  dt*c1*v],
                          [0, 0,  1]])

        else:
            r = v/w
            G = np.array([[1, 0, -r*c1 + r*c12],
                          [0, 1,  -r*s1 + r*s12],
                          [0, 0,  1]])
        return G


    # @staticmethod
    # def get_estimated_landmark_position(vertices):
    #     lm_pos = []
    #     vertices_lm = []
    #     for v in vertices:
    #         if v.type == Vertex.LANDMARK:
    #             lm_pos.append((v.pose[0, 0], v.pose[1, 0]))
    #             vertices_lm.append(v)
    #     return lm_pos, vertices_lm

    @staticmethod
    def motion_model(x, u, dt):
        """
        Noise-free motion model method
        :param x: The robot's pose
        :param u: Motion command as a tuple of translational and angular velocities
        :param dt: (Discrete) Time for which the motion command is executed
        :return: Resulting robot's pose
        """
        #print(u.T)
        # sample_noise = lambda mu, std : np.random.normal(mu, std)
        v, w = u[0, 0], u[1, 0]
        s1, s12 = sin(x[2, 0]),  sin(x[2, 0] + dt*w)
        c1, c12 = cos(x[2, 0]),  cos(x[2, 0] + dt*w)

        # No angular velocity means following a straight line
        if w == 0:
            B = np.array([[dt * c1 * v],
                          [dt * s1 * v],
                          [0.0]])
        # Otherwise the robot follows a circular arc
        else:
            r = v/w
            B = np.array([[-r*s1 + r*s12],
                          [r*c1-r*c12],
                          [w*dt]])
        x = x + B
        x[2] = atan2(sin(x[2]), cos(x[2]))
        return x


    @staticmethod
    def calc_landmark_position(x, z):
        """
        Returns the measured landmark position
        :param x: The robots pose (or combined state vector, only matters that first three elements are robot pose)
        :param z: Measurement, represented as tuple of measured distance and measured angle
        :return: Measured landmark position
        """
        lm = np.zeros((2, 1))
        lm[0, 0] = x[0, 0] + z[0] * cos(z[1] + x[2, 0])
        lm[1, 0] = x[1, 0] + z[0] * sin(z[1] + x[2, 0])
        return lm


    @staticmethod
    def diff_to_uni(v_l, v_r, width):
        v = (v_r + v_l) * 0.5
        w = (v_r - v_l) / width
        return v, w