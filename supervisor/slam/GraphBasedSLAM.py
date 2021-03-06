import time
from supervisor.slam.graph.landmark_graph import LMGraph, PoseLandmarkEdge, PosePoseEdge, PoseVertex, LandmarkVertex
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle
from models.Pose import Pose
import numpy as np
from math import *

class GraphBasedSLAM(Slam):

    def __init__(self, supervisor_interface, slam_cfg, step_time, callback = None):
        """
        Initializes an object of the GraphBasedSLAM class
        :param supervisor_interface: The interface to interact with the robot supervisor
        :param slam_cfg: The configuration for the SLAM algorithm
        :param step_time: The discrete time that a single simulation cycle increments
        :param callback: callback function
        """
        # Bind the supervisor interface
        self.supervisor = supervisor_interface
        self.draw_trajectory = slam_cfg["graph_based_slam"]["draw_trajectory"]
        # Extract relevant configurations
        self.dt = step_time
        self.robot_state_size = slam_cfg["robot_state_size"]
        self.sensor_noise = np.diag([slam_cfg["graph_based_slam"]["sensor_noise"]["x"],
                                     slam_cfg["graph_based_slam"]["sensor_noise"]["y"]])**2
        self.motion_noise = np.diag([slam_cfg["graph_based_slam"]["motion_noise"]["x"],
                                     slam_cfg["graph_based_slam"]["motion_noise"]["y"],
                                     np.deg2rad(slam_cfg["graph_based_slam"]["motion_noise"]["theta"])]) ** 2
        self.min_distance_threshold = slam_cfg["graph_based_slam"]["distance_threshold"]
        self.frontend_pose_density = slam_cfg["graph_based_slam"]["frontend_pose_density"]
        self.frontend_interval = slam_cfg["graph_based_slam"]['frontend_interval']   # the timestep interval of executing the frontend part.
        self.num_fixed_vertices = slam_cfg["graph_based_slam"]["num_fixed_vertexes"]
        # solver
        self.solver = slam_cfg["graph_based_slam"]['solver'].lower()
        # the current robot pose and its uncertainty
        self.mu = np.zeros((self.robot_state_size, 1))
        self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size)) # The state covariance, initially set to absolute certainty of the initial robot pose
        self.step_counter = 0
        self.graph = LMGraph()

        self.flg_optim = False # determines whether the graph should be optimized in a simulation cycles.
        self.callback = callback # a callback function

        # add the first node to the graph
        vertex1 = PoseVertex(self.mu, np.eye(3))
        self.graph.add_vertex(vertex1)
      
        self.counter = 0


    def get_estimated_pose(self):
        """
        Returns the estimated pose of the robot
        """
        return Pose(self.mu[0, 0], self.mu[1, 0], self.mu[2, 0])

    def get_estimated_trajectory(self):
        """
        Returns hte estimated trajectory of the robot.
        """
        return [(v.pose[0,0], v.pose[1,0])  \
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
        return [(v.pose[0, 0], v.pose[1, 0], v.landmark_id) \
                for v in self.graph.get_estimated_landmark_vertices()]

    def plot_graph(self):
        """
        plot the graph estimated by the Graph-based SLAM
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
        self.counter += 1
        start_time = time.time()

        self.mu = self.motion_model(self.mu, u, self.dt) # Calculate next step
        J = self.jaco_motion_model(self.mu, u, self.dt)  # Calculate Jacobian matrix
        self.Sigma = J @ self.Sigma @ J.T + self.motion_noise  # Update covariance matrix, propagate it while robot is moving
        self.step_counter += 1

        """        Update the Graph         """
        if self.step_counter % self.frontend_interval == 0: # create a vertex every n steps
            self.__front_end(z)
            num_vertices = len(self.graph.vertices) #  #vertices = #poses + #landmarks
            if num_vertices > 0 and self.flg_optim == True:
                self.flg_optim = False
                self.__back_end()

        if self.callback is not None:
            self.callback(str(self), time.time() - start_time)  # record time used for updating

    def __front_end(self, z):
        """
        Front end part of the Graph_based SLAM where a graph is constructed and growing while robot's moving.
        :param z: list of range-bearing measurement, a single item is (distance, angle, landmark_id) related to robot
        """

        """    calculate the next vertex of poses     """
        vertex1 = self.graph.get_last_pose_vertex() # the previous vertex
        vertex2 = PoseVertex(self.mu, self.Sigma)  # the current vertex
        distance = (vertex1.pose[0, 0] - vertex2.pose[0, 0])**2 + (vertex1.pose[1, 0] - vertex2.pose[1, 0])**2
        if distance < self.frontend_pose_density**2:  # keep the vertex density not high
            return

        self.graph.add_vertex(vertex2) # add a new pose-vertex in the graph
        """     calculate landmark vertices    """
        for i, zi in enumerate(z): # zi = [x, y, id].T
            lm_id = zi[2]  # the identifier of a detected landmark.
            # Only execute if sensor has observed landmark
            if not self.supervisor.proximity_sensor_positive_detections()[i] \
                or lm_id == -1: # not a feature
                continue
            pos_lm = self.calc_landmark_position(self.mu, zi) # calculate x-y position from range-bearing in world coordinate
            """   Data Association  """
            vertex3 = self.__data_association(zi) # Vertex3 represents an estimated landmark
            if vertex3 == None:
                # Detect a new landmark: this landmark has not been detected in the past
                vertex3 = LandmarkVertex(pos_lm, self.sensor_noise, lm_id) # create a new vertex representing a landmark
                self.graph.add_vertex(vertex3) # add the landmark-vertex to the graph
            else:
                # 1. The robot is revisiting a previous seen landmark,
                # 2. Calculate the distance between the estimated landmark via slam and that via actual measurement,
                # if the robot is revisiting a landmark, and the distance is larger than the threshold (large error),
                #       start backend to correct the inconsistency between the evaluation and the measurement.
                sq_distance = (pos_lm[0, 0] -  vertex3.pose[0, 0])**2 + (pos_lm[1, 0] -  vertex3.pose[1, 0])**2
                if sq_distance > self.min_distance_threshold**2 and self.flg_optim == False:
                    self.flg_optim = True

            # Create an edge connecting a pose and a landmark.
            measurement, info = PoseLandmarkEdge.encode_measurement(zi, self.sensor_noise) # create a spatial constraint.
            self.graph.add_edge(vertex2, vertex3, measurement, info) # add an PoseLandmarkEdge edge

        """      calculate pose-pose edge       """
        # Create an edge connecting two poses
        measurement, info = PosePoseEdge.encode_measurement(vertex1.pose, vertex2.pose, self.motion_noise) # create a spatial constraint.
        self.graph.add_edge(vertex1, vertex2, measurement, info)


    def __back_end(self):
        """
        Back end part of the Graph based slam where the graph optimization is executed.
        """
        fix_hessian = 0
        for i in range(self.num_fixed_vertices): # fix the first 20 vertices
            fix_hessian += self.graph.vertices[i].dim

        self.graph.graph_optimization(number_fix=fix_hessian, damp_factor=5, solver=self.solver)
        last_vertex = self.graph.get_last_pose_vertex()
        self.mu = np.copy(last_vertex.pose)  # update current state
        self.Sigma = np.copy(last_vertex.sigma)

    def __data_association(self, zi):
        """
        Associates the measurement to a landmark using the landmark identifiers.
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
                break
        return vertex

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


    def __str__(self):
        return "Graph-based SLAM"