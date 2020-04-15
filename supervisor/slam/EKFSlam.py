"""
EKF SLAM
Based on implementation of Atsushi Sakai (https://github.com/AtsushiSakai/PythonRobotics),
which I have also made contributions to, see pull requests #255, #258 and #305
"""

import numpy as np
from math import *
from models.Pose import Pose

# EKF state covariance
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle

"""
The sensor noise, empirically chosen. 
The first value is the standard deviation of the measured distance.
The second value is the standard deviation of the measured angle.
"""
sensor_noise = np.diag([0.2, np.deg2rad(30)]) ** 2
"""
The motion noise, empirically chosen. 
The first value is the standard deviation of the robot's x-coordinate after executing a motion command.
The second value is the standard deviation of the robot's y-coordinate after executing a motion command.
The third value is the standard deviation of the robot's angle after executing a motion command.
"""
motion_noise = np.diag([0.005, 0.005, np.deg2rad(1)]) ** 2

STATE_SIZE = 3  # State size [x,y,theta]
LM_SIZE = 2  # LM state size [x,y]


class EKFSlam(Slam):

    def __init__(self, supervisor_interface, slam_cfg, step_time):
        """
        Initializes an object of the EKFSlam class
        :param supervisor_interface: The interface to interact with the robot supervisor
        :param slam_cfg: The configuration for the SLAM algorithm
        :param step_time: The discrete time that a single simulation cycle increments
        """
        # bind the supervisor
        self.supervisor = supervisor_interface

        # Extract relevant configurations
        self.dt = step_time
        self.distance_threshold = slam_cfg["ekf_slam"]["distance_threshold"]
        self.robot_state_size = slam_cfg["robot_state_size"]
        self.landmark_state_size = slam_cfg["landmark_state_size"]

        # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
        self.mu = np.zeros((STATE_SIZE, 1))
        #
        self.Sigma = np.zeros((STATE_SIZE, STATE_SIZE))

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
        return [(x, y) for (x, y) in zip(self.mu[STATE_SIZE::2], self.mu[STATE_SIZE + 1::2])]

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
        :param z: List of sensor measurements. A single measurement is a tuple of measured distance and measured angle.
        """
        self.prediction_step(u)
        self.correction_step(z)

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
        self.Sigma[0:S, 0:S] = G.T @ self.Sigma[0:S, 0:S] @ G + motion_noise

    def correction_step(self, z):
        """
        Update the predicted state and uncertainty using the sensor measurements.
        :param z: List of sensor measurements. A single measurement is a tuple of measured distance and measured angle.
        """
        # Iterate through all sensor readings
        for i, measurement in enumerate(z):
            # Only execute if sensor observed landmark
            if not self.supervisor.proximity_sensor_positive_detections()[i]:
                continue
            lm_id = self.data_association(self.mu, self.Sigma, measurement)
            nLM = self.get_n_lm(self.mu)
            if lm_id == nLM:  # If the landmark is new
                self.add_new_landmark(measurement)
            lm = self.get_landmark_position(self.mu, lm_id)
            innovation, Psi, H = self.calc_innovation(lm, self.mu, self.Sigma, measurement, lm_id)

            K = (self.Sigma @ H.T) @ np.linalg.inv(Psi)
            self.mu += K @ innovation
            # Normalize robot angle so it is between -pi and pi
            self.mu[2] = normalize_angle(self.mu[2])
            self.Sigma = (np.identity(len(self.mu)) - (K @ H)) @ self.Sigma

    def data_association(self, mu, Sigma, measurement):
        """
        Associates the measurement to a landmark using the Mahalanobis distance
        :param mu: Combined state vector
        :param Sigma: Covariance matrix
        :param measurement: Tuple of measured distance and measured angle
        :return: The id of the landmark that is associated to the measurement
        """
        nLM = self.get_n_lm(mu)

        mdist = []

        for i in range(nLM):
            lm = self.get_landmark_position(mu, i)
            innovation, S, H = self.calc_innovation(lm, mu, Sigma, measurement, i)
            distance = innovation.T @ np.linalg.inv(S) @ innovation
            mdist.append(distance)

        mdist.append(self.distance_threshold)  # new landmark
        minid = mdist.index(min(mdist))
        return minid

    def add_new_landmark(self, measurement):
        """
        Adds a new landmark.
        State vector is extended using the measured position.
        Covariance is extended using an Identity matrix for the new landmark.
        :param measurement: Tuple of measured distance and measured angle
        """
        landmark_position = self.calc_landmark_position(self.mu, measurement)
        # Extend state and covariance matrix
        xEstTemp = np.vstack((self.mu, landmark_position))
        self.Sigma = np.vstack((np.hstack((self.Sigma, np.zeros((len(self.mu), LM_SIZE)))),
                                np.hstack((np.zeros((LM_SIZE, len(self.mu))), np.identity(LM_SIZE)))))
        self.mu = xEstTemp

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

    @staticmethod
    def jacob_motion(x, u, dt):
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
                           [0, 0, 0]])
        else:
            G = np.array([[0, 0, u[0, 0] / u[1, 0] * (cos(x[2, 0] + dt * u[1, 0]) - cos(x[2, 0]))],
                          [0, 0, u[0, 0] / u[1, 0] * (sin(x[2, 0] + dt * u[1, 0]) - sin(x[2, 0]))],
                          [0, 0, 0]])

        G = np.identity(STATE_SIZE) + G
        return G

    @staticmethod
    def jacob_sensor(q, delta, nLM, i):
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
        lm = np.zeros((2, 1))
        lm[0, 0] = x[0, 0] + z[0] * cos(z[1] + x[2, 0])
        lm[1, 0] = x[1, 0] + z[0] * sin(z[1] + x[2, 0])
        return lm

    @staticmethod
    def get_n_lm(x):
        n = int((len(x) - STATE_SIZE) / LM_SIZE)
        return n

    def calc_innovation(self, lm, xEst, PEst, z, LMid):
        delta = lm - xEst[0:2]
        q = (delta.T @ delta)[0, 0]
        zangle = atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
        expected_measurement = np.array([[sqrt(q), normalize_angle(zangle)]])
        innovation = (z - expected_measurement).T
        innovation[1] = normalize_angle(innovation[1])
        H = self.jacob_sensor(q, delta, self.get_n_lm(xEst), LMid)
        Psi = H @ PEst @ H.T + sensor_noise

        return innovation, Psi, H

    @staticmethod
    def get_landmark_position(x, ind):
        lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
        return lm

