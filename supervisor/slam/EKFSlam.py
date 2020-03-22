import numpy as np
from math import *
from models.Pose import Pose

# EKF state covariance
sensor_noise = np.diag([0.2, np.deg2rad(30)]) ** 2
motion_noise = np.diag([0.05, 0.05, np.deg2rad(1)]) ** 2

STATE_SIZE = 3  # State size [x,y,theta]
LM_SIZE = 2  # LM state size [x,y]


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))
    zp[0, 0] = x[0, 0] + z[0] * cos(z[1] + x[2, 0])
    zp[1, 0] = x[1, 0] + z[0] * sin(z[1] + x[2, 0])
    return zp


# return number of obeserved landmarks
def get_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_sensor(q, delta, x, i):
    sq = sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], -q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = get_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * (nLM - i)))))

    F = np.vstack((F1, F2))

    H = G @ F

    # H is just G, but with many extra [0, 0] columns for all landmarks with id different i, and the last 2 columns of G at the position of i

    return H


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    zangle = atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[sqrt(q), pi_2_pi(zangle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_sensor(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + sensor_noise

    return y, S, H


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm


def search_correspond_landmark_id(xAug, PAug, zi, distance_threshold):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = get_n_lm(xAug)

    mdist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        distance = y.T @ np.linalg.inv(S) @ y
        mdist.append(distance)

    mdist.append(distance_threshold)  # new landmark
    minid = mdist.index(min(mdist))
    return minid


class EKFSlam:

    def __init__(self, supervisor_interface, obs_radius, slam_cfg, step_time):
        # bind the supervisor
        self.supervisor = supervisor_interface

        # Extract relevant configurations
        self.dt = step_time
        self.obs_radius = obs_radius
        self.distance_threshold = slam_cfg["distance_threshold"]

        self.xEst = np.zeros((STATE_SIZE, 1))
        self.PEst = np.zeros((STATE_SIZE, STATE_SIZE))

    def execute(self, u, z):
        # Predict
        S = STATE_SIZE
        self.xEst[0:S] = self.motion_model(self.xEst[0:S], u, self.dt)
        G = self.jacob_motion(self.xEst[0:S], u, self.dt)
        self.PEst[0:S, 0:S] = G.T @ self.PEst[0:S, 0:S] @ G + motion_noise
        # Update
        assert len(z) == len(self.supervisor.proximity_sensor_placements())
        z = zip(z, [pose.theta for pose in self.supervisor.proximity_sensor_placements()])
        for iz, (distance, theta) in enumerate(z):
            if distance >= self.supervisor.proximity_sensor_max_range() - 0.01:  # only execute if landmark is observed
                continue
            distance += self.obs_radius  # TODO: Rethink this
            minid = search_correspond_landmark_id(self.xEst, self.PEst, [distance, theta], self.distance_threshold)

            nLM = get_n_lm(self.xEst)
            if minid == nLM:   # If the landmark is new
                # Extend state and covariance matrix
                landmark_position = calc_landmark_position(self.xEst, [distance, theta])
                xAug = np.vstack((self.xEst, landmark_position))
                PAug = np.vstack((np.hstack((self.PEst, np.zeros((len(self.xEst), LM_SIZE)))),
                                  np.hstack((np.zeros((LM_SIZE, len(self.xEst))), np.identity(LM_SIZE)))))
                self.xEst = xAug
                self.PEst = PAug

            lm = get_landmark_position_from_state(self.xEst, minid)
            y, S, H = calc_innovation(lm, self.xEst, self.PEst, [distance, theta], minid)

            K = (self.PEst @ H.T) @ np.linalg.inv(S)
            #print("Kalman gain:", K)
            #print("Correction:", K @ y)
            self.xEst = self.xEst + (K @ y)
            self.PEst = (np.identity(len(self.xEst)) - (K @ H)) @ self.PEst

        self.xEst[2] = pi_2_pi(self.xEst[2])

    def get_estimated_pose(self):
        return Pose(self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0])

    def get_covariances(self):
        return self.PEst

    def get_landmarks(self):
        return [(x, y) for (x, y) in zip(self.xEst[STATE_SIZE::2], self.xEst[STATE_SIZE+1::2])]

    # The motion model for a motion command u = (velocity, angular velocity)
    def motion_model(self, x, u, dt):
        if u[1, 0] == 0:
            B = np.array([[dt * cos(x[2, 0]) * u[0, 0]],
                          [dt * sin(x[2, 0]) * u[0, 0]],
                          [0.0]])
        else:
            B = np.array([[u[0, 0] / u[1, 0] * (sin(x[2, 0] + dt * u[1, 0]) - sin(x[2, 0]))],
                          [u[0, 0] / u[1, 0] * (-cos(x[2, 0] + dt * u[1, 0]) + cos(x[2, 0]))],
                          [u[1, 0] * dt]])
        res = x + B
        res[2] = pi_2_pi(res[2])
        return res

    def jacob_motion(self, x, u, dt):
        if u[1, 0] == 0:
            jF = np.array([[0, 0, -dt * u[0] * sin(x[2, 0])],
                           [0, 0, dt * u[0] * cos(x[2, 0])],
                           [0, 0, 0]])
        else:
            jF = np.array([[0, 0, u[0, 0] / u[1, 0] * (cos(x[2, 0] + dt * u[1, 0]) - cos(x[2, 0]))],
                          [0, 0, u[0, 0] / u[1, 0] * (sin(x[2, 0] + dt * u[1, 0]) - sin(x[2, 0]))],
                          [0, 0, 0]])

        G = np.identity(STATE_SIZE) + jF
        return G



