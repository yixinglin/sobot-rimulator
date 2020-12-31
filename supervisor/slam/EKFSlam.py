"""
EKF SLAM
Based on implementation of Atsushi Sakai (https://github.com/AtsushiSakai/PythonRobotics),
which I have also made contributions to, see pull requests #255, #258 and #305
"""
import time
import numpy as np
from math import *
from models.Pose import Pose

from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle
from itertools import cycle

class EKFSlam(Slam):

    def __init__(self, supervisor_interface, slam_cfg, step_time, callback = None):
        """
        Initializes an object of the EKFSlam class
        :param supervisor_interface: The interface to interact with the robot supervisor
        :param slam_cfg: The configuration for the SLAM algorithm
        :param step_time: The discrete time that a single simulation cycle increments
        :param callback: callback function
        """
        # Bind the supervisor interface
        self.supervisor = supervisor_interface
        # Extract relevant configurations
        self.dt = step_time
        self.distance_threshold = slam_cfg["ekf_slam"]["distance_threshold"]
        self.robot_state_size = slam_cfg["robot_state_size"]
        self.landmark_state_size = slam_cfg["landmark_state_size"]
        self.sensor_noise = np.diag([slam_cfg["ekf_slam"]["sensor_noise"]["detected_distance"],
                                     np.deg2rad(slam_cfg["ekf_slam"]["sensor_noise"]["detected_angle"])]) ** 2
        self.motion_noise = np.diag([slam_cfg["ekf_slam"]["motion_noise"]["x"],
                                     slam_cfg["ekf_slam"]["motion_noise"]["y"],
                                     np.deg2rad(slam_cfg["ekf_slam"]["motion_noise"]["theta"])]) ** 2
        self.landmark_correspondence_given = slam_cfg["feature_detector"]
        # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
        self.mu = np.zeros((self.robot_state_size, 1))
        # The state covariance, initially set to absolute certainty of the initial robot pose
        self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size))
        # The list of landmark IDs.
        self.landmark_id = list()

        self.callback = callback

    def get_estimated_pose(self):
        """
        Returns the estimated robot pose by retrieving the first three elements of the combined state vector
        :return: Estimated robot pose consisting of position and angle
        """
        return Pose(self.mu[0, 0], self.mu[1, 0], self.mu[2, 0])

    def get_landmarks(self):
        """
        Returns the estimated landmark positions
        :return: List of estimated landmark positions
        """
        return [(x[0], y[0], id) for (x, y, id) in zip(self.mu[self.robot_state_size::2], self.mu[self.robot_state_size + 1::2], cycle(self.landmark_id))]

    def get_covariances(self):
        """
        Returns the covariance matrix
        :return: Covariance matrix as a NumPy matrix
        """
        return self.Sigma

    def update(self, u, z):
        """
        Performs a full update cycle consisting of prediction and correction step
        :param u: Motion command
        :param z: List of sensor measurements. A single measurement is a tuple of measured distance, measured angle and associated index of landmark.
        """
        start_time = time.time()

        self.prediction_step(u)
        self.correction_step(z)

        if self.callback is not None:
            self.callback(str(self), time.time() - start_time) # time used for updating


    def prediction_step(self, u):
        """
        Predicts the robots location and location uncertainty after the execution of a motion command.
        The estimated landmarks remain unchanged.
        After executing this method, self.mu and self.Sigma contain the predicted state and covariance.
        :param u: Motion command
        """
        S = self.robot_state_size
        # Compute the Jacobian matrix G
        G = self.jacob_motion(self.mu[0:S], u, self.dt)
        # Predict the robots pose by executing noise-free motion
        self.mu[0:S] = self.motion_model(self.mu[0:S], u, self.dt)
        # Update the uncertainty of the robots pose using Jacobian G
        self.Sigma[0:S, 0:S] = G.T @ self.Sigma[0:S, 0:S] @ G + self.motion_noise

    def correction_step(self, z):
        """
        Update the predicted state and uncertainty using the sensor measurements.
        :param z: List of sensor measurements. A single measurement is a tuple of measured distance, measured angle and identify.
        """
        # Iterate through all sensor readings
        for i, measurement in enumerate(z):
            # Only execute if sensor observed landmark
            if not self.supervisor.proximity_sensor_positive_detections()[i]\
                or measurement[2] == -1: # not a feature
                continue
            if not self.landmark_correspondence_given:
                lm_id = self.data_association(self.mu, self.Sigma, measurement[0:2])
            else:
                lm_id = self.data_association_v2(self.mu, measurement[2])
            nLM = self.get_n_lm(self.mu)
            if lm_id == nLM:  # If the landmark is new
                self.add_new_landmark(measurement[0:2], measurement[2])
            lm = self.get_landmark_position(self.mu, lm_id)
            innovation, Psi, H = self.calc_innovation(lm, self.mu, self.Sigma, measurement[0:2], lm_id)

            K = (self.Sigma @ H.T) @ np.linalg.inv(Psi)
            self.mu += K @ innovation
            # Normalize robot angle so it is between -pi and pi
            self.mu[2] = normalize_angle(self.mu[2])
            self.Sigma = (np.identity(len(self.mu)) - (K @ H)) @ self.Sigma

    def data_association(self, mu, Sigma, measurement):
        """
        Associates the measurement to a landmark using the Mahalanobis distance.
        The innovation, uncertainty and Jacobian are however not returned
        and need to be recalculated when performing the EKF update.
        :param mu: Combined state vector
        :param Sigma: Covariance matrix
        :param measurement: Tuple of measured distance and measured angle
        :return: The id of the landmark that is associated to the measurement
        """
        nLM = self.get_n_lm(mu)
        mdist = []
        # This distance is used to skip the calculation of the Mahalanobis distance for landmarks
        # that are estimated to be far away (further than twice the maximum sensor range)
        squared_cutoff_distance = (2 * self.supervisor.proximity_sensor_max_range()) ** 2
        for i in range(nLM):
            lm = self.get_landmark_position(mu, i)
            delta = lm - mu[:2]
            # If landmark is too far away, don't bother calculating Mahalanobis distance
            if delta[0] ** 2 + delta[1] ** 2 > squared_cutoff_distance:
                mdist.append(self.distance_threshold + 1)  # Will not be considered
            else:
                innovation, Psi, H = self.calc_innovation(lm, mu, Sigma, measurement, i)
                distance = innovation.T @ np.linalg.inv(Psi) @ innovation
                mdist.append(distance)
        mdist.append(self.distance_threshold)  # new landmark
        minid = mdist.index(min(mdist))
        return minid

    def data_association_v2(self, mu, id):
        """
        Associates the identify to a landmark.
        :param particle: Particle that will be updated
        :param id: Identify of the observed landmark
        return: The landmark index in the list.
        """
        nLM = self.get_n_lm(mu)
        lm_index = nLM
        for i in range(nLM):
            if self.landmark_id[i] == id:
                lm_index = i
        return lm_index

    def add_new_landmark(self, measurement, id):
        """
        Adds a new landmark.
        State vector is extended using the measured position.
        Covariance is extended using an Identity matrix for the new landmark.
        :param measurement: Tuple of measured distance and measured angle
        """
        landmark_position = self.calc_landmark_position(self.mu, measurement)
        # Extend state and covariance matrix
        xEstTemp = np.vstack((self.mu, landmark_position))
        L = self.landmark_state_size
        self.Sigma = np.vstack((np.hstack((self.Sigma, np.zeros((len(self.mu), L)))),
                                np.hstack((np.zeros((L, len(self.mu))), np.identity(L)))))
        self.mu = xEstTemp
        self.landmark_id.append(id)

    @staticmethod
    def motion_model(x, u, dt):
        """
        Noise-free motion model method
        :param x: The robot's pose
        :param u: Motion command as a tuple of translational and angular velocities
        :param dt: (Discrete) Time for which the motion command is executed
        :return: Resulting robot's pose
        """
        # No angular velocity means following a straight line
        if u[1, 0] == 0:
            B = np.array([[dt * cos(x[2, 0]) * u[0, 0]],
                          [dt * sin(x[2, 0]) * u[0, 0]],
                          [0.0]])
        # Otherwise the robot follows a circular arc
        else:
            B = np.array([[u[0, 0] / u[1, 0] * (sin(x[2, 0] + dt * u[1, 0]) - sin(x[2, 0]))],
                          [u[0, 0] / u[1, 0] * (-cos(x[2, 0] + dt * u[1, 0]) + cos(x[2, 0]))],
                          [u[1, 0] * dt]])
        res = x + B
        res[2] = normalize_angle(res[2])
        return res

    def jacob_motion(self, x, u, dt):
        """
        Returns the Jacobian matrix of the motion model
        :param x: The robot's pose
        :param u: Motion command as a tuple of translational and angular velocities
        :param dt: (Discrete) Time for which the motion command is executed
        :return: Jacobian matrix of the motion model
        """
        if u[1, 0] == 0:
            G = np.array([[0, 0, -dt * u[0] * sin(x[2, 0])],
                           [0, 0, dt * u[0] * cos(x[2, 0])],
                           [0, 0, 0]], dtype=np.float)
        else:
            G = np.array([[0, 0, u[0, 0] / u[1, 0] * (cos(x[2, 0] + dt * u[1, 0]) - cos(x[2, 0]))],
                          [0, 0, u[0, 0] / u[1, 0] * (sin(x[2, 0] + dt * u[1, 0]) - sin(x[2, 0]))],
                          [0, 0, 0]], dtype=np.float)

        G = np.identity(self.robot_state_size) + G
        return G

    @staticmethod
    def jacob_sensor(q, delta, nLM, i):
        """
        Computes the Jacobian of the sensor model
        :param q: squared distance of the expected measurement
        :param delta: vector of the expected measurement (estimated landmark position - robot position)
        :param nLM: Number of observed landmarks
        :param i: Index of the observed landmark
        :return: Jacobian of measurement
        """
        sq = sqrt(q)
        H = np.zeros((2, 3 + nLM * 2))
        # Setting the values dependent on the robots pose
        H[:, :3] = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0],
                             [delta[1, 0], - delta[0, 0], -q]])
        # Setting the values dependent on the landmark location
        H[:, 3 + i * 2: 3 + (i+1) * 2] = np.array([[sq * delta[0, 0], sq * delta[1, 0]],
                                                   [- delta[1, 0], delta[0, 0]]])
        H = H / q
        return H

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

    def get_n_lm(self, mu):
        """
        Returns number of observed landmarks
        :param mu: Combined state vector
        :return: Number of observed landmarks
        """
        n = int((len(mu) - self.robot_state_size) / self.landmark_state_size)
        return n

    def calc_innovation(self, lm, mu, Sigma, z, LMid):
        """
        Calculates the innovation, uncertainty and Jacobian
        :param lm: Position of observed landmark
        :param mu: Combined state vector
        :param Sigma: Covariance matrix
        :param z: Measurement, consisting of tuple of measured distance and measured angle
        :param LMid: Id of the observed landmark
        :return: The innovation, the uncertainty of the measurement and the Jacobian
        """
        delta = lm - mu[0:2]
        q = (delta.T @ delta)[0, 0]
        zangle = atan2(delta[1, 0], delta[0, 0]) - mu[2, 0]
        expected_measurement = np.array([[sqrt(q), normalize_angle(zangle)]])
        innovation = (z - expected_measurement).T
        innovation[1] = normalize_angle(innovation[1])
        H = self.jacob_sensor(q, delta, self.get_n_lm(mu), LMid)
        Psi = H @ Sigma @ H.T + self.sensor_noise

        return innovation, Psi, H

    def get_landmark_position(self, mu, i):
        """
        Returns the landmark with specified index
        :param mu: Combined state vector
        :param i: Index of landmark
        :return: The (estimated) position of the landmark with index i
        """
        R = self.robot_state_size
        L = self.landmark_state_size
        lm = mu[R + L * i: R + L * (i + 1), :]
        return lm

    def __str__(self):
        return "EKF SLAM"
