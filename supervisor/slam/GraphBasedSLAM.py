from threading import Thread, Lock
from supervisor.slam.graph.landmark_graph import *
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle
from models.Pose import Pose
import numpy as np
from math import *

optimize_allowed = True

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
        self.sensor_noise = np.diag([slam_cfg["graph_based_slam"]["sensor_noise"]["x"],
                                     slam_cfg["graph_based_slam"]["sensor_noise"]["y"]])**2
        self.motion_noise = np.diag([slam_cfg["graph_based_slam"]["motion_noise"]["x"],
                                     slam_cfg["graph_based_slam"]["motion_noise"]["y"],
                                     np.deg2rad(slam_cfg["graph_based_slam"]["motion_noise"]["theta"])]) ** 2
        self.min_distance_threshold = slam_cfg["graph_based_slam"]["distance_threshold"]
        self.optimization_interval = slam_cfg["graph_based_slam"]['optimization_interval'] # the number interval of pose-vertices added that the graph optimization is executed.
        self.frontend_interval = slam_cfg["graph_based_slam"]['frontend_interval']   # the timestep interval of executing the frontend part.

        # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
        self.mu = np.zeros((self.robot_state_size, 1))
        self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size)) # The state covariance, initially set to absolute certainty of the initial robot pose
        self.step_counter = 0
        self.max_range = self.supervisor.proximity_sensor_max_range()
        self.min_range = self.supervisor.proximity_sensor_min_range()
        self.graph = LMGraph()
        self.odom_pose = self.__reset_odometry_measurement() # accumulative odometry estimation from wheel encoders
        self.fix_hessian = 0 # number of fixed vertices while the graph optimzation.
        self.__init_first_step()  # initialize the first step

    def __init_first_step(self):
        """    add the initial robot pose as the first pose-vertex     """
        vertex1 = PoseVertex(self.mu, np.eye(3))
        self.graph.add_vertex(vertex1)
        self.fix_hessian += 3 # fix this vertex

    def get_estimated_pose(self):
        """
        Returns the estimated pose of the robot
        """
        return Pose(self.mu[0, 0], self.mu[1, 0], self.mu[2, 0])

    def get_estimated_trajectory(self):
        return [ (v.pose[0,0], v.pose[1,0])  \
                 for v in self.graph.get_estimated_pose_vertices()]

    def get_hessian(self):
        """
        Return the hessian matrix H and the information vector b
        :return:  H, b
            H: the hessian matrix
            b: the information vector
        """
        H, b = self.graph.get_hessian()
        return H.toarray(), b

    def get_landmarks(self):
        """
        Returns the estimated landmark positions
        """
        return [(v.pose[0, 0], v.pose[1, 0]) \
                for v in self.graph.get_estimated_landmark_vertices() ]

    def plot_graph(self):
        """
        plot the graph estimated by slam
        """
        self.graph.draw()

    def __reset_odometry_measurement(self):
        return np.copy(self.mu)

    def update(self, u, z):
        """
        Executes an update cycle of the SLAM algorithm
        :param u: Motion command, A single item is a vector of [[translational_velocity], [angular_velocity]]
        :param z: List of measurements. A single item is a tuple of (range, angle, landmark_id)
        """
        self.mu = self.motion_model(self.mu, u, self.dt) # Calculate next step
        J = self.jaco_motion_model(self.mu, u, self.dt)  # Calculate Jacobian matrix
        self.Sigma = J @ self.Sigma @ J.T + self.motion_noise  # Update covariance matrix, propagate it while robot is moving
        self.step_counter += 1

        """        Update the Graph         """
        if self.step_counter % self.frontend_interval == 0: # create a vertex each n steps
            self.__front_end(z)
            num_poses, _ = self.graph.count_vertices()

            if num_poses % self.optimization_interval == 0 \
                    and num_poses > 0 or num_poses == self.optimization_interval//2:
                self.__back_end() # graph optimization

    def __front_end(self, z):
        """
        Front end part of the Graph_based SLAM where a graph is built and growing as robot's moving.
        :param z: list of range-bearing measurement, a single item is (distance, angle, landmark_id) related to robot
        """

        """    calculate the next vertex of poses     """
        vertex1 = self.graph.get_last_pose_vertex() # a previous vertex
        vertex2 = PoseVertex(self.mu, self.Sigma)  # a current vertex
        self.graph.add_vertex(vertex2)

        """     calculate landmark vertices    """
        for i, zi in enumerate(z):
            # Only execute if sensor has observed landmark
            if not self.supervisor.proximity_sensor_positive_detections()[i]:
                continue
            pos_lm = self.calc_landmark_position(self.mu, zi) # calculate x-y position from range-bearing in world coordinate

            """   Data Association  """
            vertex3 = self.__data_association(zi) # vertex3 is a vertex of landmark
            if vertex3 == None: # Detect a new landmark, this landmark has not been detected in the past
                vertex3 = LandmarkVertex(pos_lm, self.sensor_noise, zi[2])
                self.graph.add_vertex(vertex3)

            measurement, info = PoseLandmarkEdge.encode_measurement(zi, self.sensor_noise)
            self.graph.add_edge(vertex2, vertex3, measurement, info) # add an PoseLandmarkEdge edge

        """      calculate pose-pose edge       """
        measurement, info = PosePoseEdge.encode_measurement(vertex1.pose, vertex2.pose, self.motion_noise)
        self.graph.add_edge(vertex1, vertex2, measurement, info)

    def __back_end(self):
        """
        Back end part of the Graph based slam where the graph optimization is executed.
        """
        self.graph.graph_optimization(number_fix=self.fix_hessian, damp_factor=1)
        last_vertex = self.graph.get_last_pose_vertex()
        self.mu = np.copy(last_vertex.pose)  # update current state
        self.Sigma = np.copy(last_vertex.sigma)

    def __calc_range_bearing_delta(self, x, lm, z):
        delta = lm - x[0:2, :]
        range = np.linalg.norm(delta, axis=0)
        phi = np.arctan2(delta[1, :], delta[0, :])
        ext_z = np.vstack((range, phi))
        err = ext_z - np.array(z).reshape((2,1))
        err[1, :] = np.arctan2(np.sin(err[1, :]), np.cos(err[1, :]))
        return err

    def __data_association(self, zi):
        """
        Associates the measurement to a landmark using the landmark id.
        :param zi: (distance, angle, landmark_id)
        return
            vertex: a vertex object with the same id containing in zi
        """
        lm_id = zi[2] # landmark id
        landmark_vertices = self.graph.get_estimated_landmark_vertices()
        vertex = None
        for v in landmark_vertices:
            if lm_id == v.landmark_id:
                vertex = v
        return vertex

    def get_estimated_landmark_position(self):
        lm_pos = []
        vertices_lm = self.graph.get_estimated_landmark_vertices()
        for v in vertices_lm:
            lm_pos.append((v.pose[0, 0], v.pose[1, 0]))
        return lm_pos, vertices_lm

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



    @staticmethod
    def motion_model(x, u, dt):
        """
        Noise-free motion model method
        :param x: The robot's pose
        :param u: Motion command as a tuple of translational and angular velocities
        :param dt: (Discrete) Time for which the motion command is executed
        :return: Resulting robot's pose
        """
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
        x[2] = normalize_angle(x[2])
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
        """
        :param v_l: Translational velocity of the left wheel
        :param v_r: Translational velocity of the right wheel
        :return v, w
                Translational and angular velocities
        """
        v = (v_r + v_l) * 0.5
        w = (v_r - v_l) / width
        return v, w