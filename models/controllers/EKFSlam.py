import numpy as np
from math import *
from models.pose import Pose

# EKF state covariance
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2

STATE_SIZE = 3  # State size [x,y,theta]
LM_SIZE = 2  # LM state size [x,y]
M_DIST_TH = 0.1  # Threshold of Mahalanobis distance for data association.


def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))
    zp[0, 0] = x[0, 0] + z[0] * cos(x[2, 0] + z[1])  # TODO: See if adding a constant to z[0] helps to compensate for the robot size
    zp[1, 0] = x[1, 0] + z[0] * sin(x[2, 0] + z[1])
    return zp


# return number of obeserved landmarks
def get_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE)
    return n


def jacob_h(q, delta, x, i):
    sq = sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - 1.0, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = get_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    zangle = atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[sqrt(q), pi_2_pi(zangle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]
    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
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
        print("Landmark with id", i, "has distance", distance)

    mdist.append(M_DIST_TH)  # new landmark
    minid = mdist.index(min(mdist))
    return minid


class EKFSlam:

    def __init__(self, supervisor_interface, step_time):
        # bind the supervisor
        self.supervisor = supervisor_interface

        self.dt = step_time

        self.xEst = np.zeros((STATE_SIZE, 1))
        self.PEst = np.identity(STATE_SIZE)

    def ekf_slam(self, u, z):
        # Predict
        S = STATE_SIZE
        self.xEst[0:S] = self.motion_model(self.xEst[0:S], u)
        G, Fx = self.jacob_motion(self.xEst[0:S], u)
        self.PEst[0:S, 0:S] = G.T * self.PEst[0:S, 0:S] * G + Fx.T * Cx * Fx
        initP = np.eye(2)
        #print("Current Pose Estimate", self.xEst[0], " ", self.xEst[1])
        # Update
        assert len(z) == len(self.supervisor.proximity_sensor_placements())
        z = zip(z, [pose.theta for pose in self.supervisor.proximity_sensor_placements()])
        for iz, (distance, theta) in enumerate(z):
            if distance >= self.supervisor.proximity_sensor_max_range() - 0.01:  # only execute if landmark is observed
                continue
            minid = search_correspond_landmark_id(self.xEst, self.PEst, [distance, theta])

            nLM = get_n_lm(self.xEst)
            print("Seeing landmark with Id: ", minid)
            print("Number of landmarks is: ", nLM)
            if minid == nLM:   # If the landmark is new
                print("New Landmark")
                # Extend state and covariance matrix
                landmark_position = calc_landmark_position(self.xEst, [distance, theta])
                xAug = np.vstack((self.xEst, landmark_position))
                PAug = np.vstack((np.hstack((self.PEst, np.zeros((len(self.xEst), LM_SIZE)))),
                                  np.hstack((np.zeros((LM_SIZE, len(self.xEst))), initP))))
                self.xEst = xAug
                self.PEst = PAug
            lm = get_landmark_position_from_state(self.xEst, minid)
            y, S, H = calc_innovation(lm, self.xEst, self.PEst, [distance, theta], minid)

            K = (self.PEst @ H.T) @ np.linalg.inv(S)
            self.xEst = self.xEst + (K @ y)
            self.PEst = (np.eye(len(self.xEst)) - (K @ H)) @ self.PEst
        #print("Updated Pose Estimate", self.xEst[0], " ", self.xEst[1])

        self.xEst[2] = pi_2_pi(self.xEst[2])

    def get_estimated_pose(self):
        return Pose(self.xEst[0, 0], self.xEst[1, 0], self.xEst[2, 0])

    def get_landmarks(self):
        return [(x, y) for (x, y) in zip(self.xEst[STATE_SIZE::2], self.xEst[STATE_SIZE+1::2])]

    # The motion model for a motion command u = (velocity, angular velocity)
    def motion_model(self, x, u):
        B = np.array([[self.dt * cos(x[2, 0]), 0],
                      [self.dt * sin(x[2, 0]), 0],
                      [0.0, self.dt]])
        res = x + (B @ u)
        return res

    def jacob_motion(self, x, u):
        Fx = np.hstack((np.identity(STATE_SIZE),
                        np.zeros((STATE_SIZE, LM_SIZE * get_n_lm(x)))))

        jF = np.array([[0, 0, -self.dt * u[0] * sin(x[2, 0])],
                       [0, 0, self.dt * u[0] * cos(x[2, 0])],
                       [0, 0, 0]])

        G = np.eye(STATE_SIZE) + Fx.transpose() * jF * Fx

        return G, Fx



