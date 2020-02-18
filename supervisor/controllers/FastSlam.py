"""
FastSLAM 1.0 example
author: Atsushi Sakai (@Atsushi_twi)
"""

from math import cos, sin, sqrt, atan2, exp, pi

import numpy as np

# Fast SLAM covariance
from models.Pose import Pose

sensor_noise = np.diag([0.2, np.deg2rad(30.0)]) ** 2
motion_noise = np.diag([0.01, 0.01]) ** 2

M_DIST_TH = 0.15  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling


class Particle:

    def __init__(self):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((0, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((0, LM_SIZE))


class FastSlam:

    def __init__(self, supervisor_interface, step_time):
        self.supervisor = supervisor_interface
        self.dt = step_time
        self.particles = [Particle() for _ in range(N_PARTICLE)]

    def fast_slam(self, u, z):
        self.particles = self.predict_particles(self.particles, u)

        self.particles = self.update_with_observation(self.particles, z)

        self.particles = self.resampling(self.particles)

        return self.particles

    def get_estimated_pose(self):
        xEst = self.calc_final_state(self.particles)
        return Pose(xEst[0, 0], xEst[1, 0], xEst[2, 0])

    def get_landmarks(self):
        lmEst = self.calc_final_landmarks(self.particles)
        n_lms = get_n_lms(lmEst)
        return [(x, y) for (x, y) in zip(lmEst[:n_lms, 0], lmEst[:n_lms, 1])]

    def normalize_weight(self, particles):
        sumw = sum([p.w for p in particles])
        try:
            for particle in particles:
                particle.w /= sumw
        except ZeroDivisionError:
            for particle in particles:
                particle.w = 1.0 / N_PARTICLE
        return particles

    def calc_final_state(self, particles):
        xEst = np.zeros((STATE_SIZE, 1))
        particles = self.normalize_weight(particles)
        for particle in particles:
            xEst[0, 0] += particle.w * particle.x
            xEst[1, 0] += particle.w * particle.y
            xEst[2, 0] += particle.w * particle.yaw
        xEst[2, 0] = self.pi_2_pi(xEst[2, 0])
        return xEst

    def calc_final_landmarks(self, particles):
        max_n_lms = max([get_n_lms(particle.lm) for particle in particles])
        lmEst = np.zeros((max_n_lms, LM_SIZE))
        particles = self.normalize_weight(particles)
        for particle in particles:
            n_lms = get_n_lms(particle.lm)
            lmEst[:n_lms] += particle.w * particle.lm
        return lmEst

    def predict_particles(self, particles, u):
        for particle in particles:
            px = np.zeros((STATE_SIZE, 1))
            px[0, 0] = particle.x
            px[1, 0] = particle.y
            px[2, 0] = particle.yaw
            u += (np.random.randn(1, 2) @ motion_noise ** 0.5).T  # TODO : Think if adding this noise makes sense
            px = self.motion_model(px, u)
            particle.x = px[0, 0]
            particle.y = px[1, 0]
            particle.yaw = px[2, 0]
        return particles

    def add_new_lm(self, particle, z):
        r = z[0]
        b = z[1]

        s = sin(self.pi_2_pi(particle.yaw + b))
        c = cos(self.pi_2_pi(particle.yaw + b))

        new_lm = np.array([particle.x + r * c, particle.y + r * s]).reshape(1, LM_SIZE)
        particle.lm = np.vstack((particle.lm, new_lm))

        # covariance
        Gz = np.array([[c, -r * s],
                       [s, r * c]])

        particle.lmP = np.vstack((particle.lmP, Gz @ sensor_noise @ Gz.T))

        return particle

    def compute_jacobians(self, particle, xf, Pf):
        dx = xf[0, 0] - particle.x
        dy = xf[1, 0] - particle.y
        d2 = dx ** 2 + dy ** 2
        d = sqrt(d2)

        zp = np.array(
            [d, self.pi_2_pi(atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

        Hv = np.array([[-dx / d, -dy / d, 0.0],
                       [dy / d2, -dx / d2, -1.0]])

        Hf = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])

        Sf = Hf @ Pf @ Hf.T + sensor_noise

        return zp, Hv, Hf, Sf

    def update_kf_with_cholesky(self, xf, Pf, v, Hf):
        PHt = Pf @ Hf.T
        S = Hf @ PHt + sensor_noise

        S = (S + S.T) * 0.5
        SChol = np.linalg.cholesky(S).T
        SCholInv = np.linalg.inv(SChol)
        W1 = PHt @ SCholInv
        W = W1 @ SCholInv.T

        x = xf + W @ v
        P = Pf - W1 @ W1.T

        return x, P

    def update_landmark(self, particle, z):
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf)

        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        xf, Pf = self.update_kf_with_cholesky(xf, Pf, dz, Hf)

        particle.lm[lm_id, :] = xf.T
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

        return particle

    def compute_weight(self, particle, z):
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf)

        dx = z[0:2].reshape(2, 1) - zp
        dx[1, 0] = self.pi_2_pi(dx[1, 0])

        try:
            invS = np.linalg.inv(Sf)
        except np.linalg.linalg.LinAlgError:
            print("singuler")
            return 1.0

        num = exp(-0.5 * dx.T @ invS @ dx)
        den = 2.0 * pi * sqrt(np.linalg.det(Sf))

        w = num / den

        return w

    def update_with_observation(self, particles, z):
        z = zip(z, [pose.theta for pose in self.supervisor.proximity_sensor_placements()])
        for (distance, theta) in z:
            if distance >= self.supervisor.proximity_sensor_max_range() - 0.01:  # only execute if landmark is observed
                continue

            for particle in particles:
                x = np.array([particle.x, particle.y, particle.yaw]).reshape(3, 1)
                minid = search_correspond_landmark_id(x, particle.lm, [distance, theta])
                nLM = get_n_lms(particle.lm)

                if minid == nLM:   # If the landmark is new
                    self.add_new_lm(particle, np.asarray([distance, theta, minid]))
                else:
                    w = self.compute_weight(particle, np.asarray([distance, theta, minid]))
                    particle.w *= w
                    self.update_landmark(particle, np.asarray([distance, theta, minid]))

        return particles

    def resampling(self, particles):
        """
        low variance re-sampling
        """

        particles = self.normalize_weight(particles)

        pw = np.array([particle.w for particle in particles])

        Neff = 1.0 / (pw @ pw.T)  # Effective particle number
        # print(Neff)

        if Neff < NTH:  # resampling
            wcum = np.cumsum(pw)
            base = np.cumsum(pw * 0.0 + 1 / N_PARTICLE) - 1 / N_PARTICLE
            resampleid = base + np.random.rand(base.shape[0]) / N_PARTICLE

            inds = []
            ind = 0
            for ip in range(N_PARTICLE):
                while (ind < wcum.shape[0] - 1) and (resampleid[ip] > wcum[ind]):
                    ind += 1
                inds.append(ind)

            tparticles = particles[:]
            for i in range(len(inds)):
                particles[i].x = tparticles[inds[i]].x
                particles[i].y = tparticles[inds[i]].y
                particles[i].yaw = tparticles[inds[i]].yaw
                particles[i].lm = tparticles[inds[i]].lm[:, :]
                particles[i].lmP = tparticles[inds[i]].lmP[:, :]
                particles[i].w = 1.0 / N_PARTICLE

        return particles

    # The motion model for a motion command u = (velocity, angular velocity)
    def motion_model(self, x, u):
        if u[1, 0] == 0:
            B = np.array([[self.dt * cos(x[2, 0]) * u[0, 0]],
                          [self.dt * sin(x[2, 0]) * u[0, 0]],
                          [0.0]])
        else:
            B = np.array([[u[0, 0] / u[1, 0] * (sin(x[2, 0] + self.dt * u[1, 0]) - sin(x[2, 0]))],
                          [u[0, 0] / u[1, 0] * (-cos(x[2, 0] + self.dt * u[1, 0]) + cos(x[2, 0]))],
                          [u[1, 0] * self.dt]])
        res = x + B
        res[2] = self.pi_2_pi(res[2])
        return res

    def pi_2_pi(self, angle):
        return (angle + pi) % (2 * pi) - pi



def search_correspond_landmark_id(x, lm, z):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = get_n_lms(lm)

    mdist = []
    measured_lm = calc_landmark_position(x, z)

    for i in range(nLM):
        lm_i = lm[i]
        delta = lm_i - measured_lm
        distance = sqrt(delta[0, 0] ** 2 + delta[0, 1] ** 2)
        mdist.append(distance)

    mdist.append(M_DIST_TH)  # new landmark
    minid = mdist.index(min(mdist))
    #
    return minid


def calc_landmark_position(x, z):
    zp = np.zeros((1, 2))
    zp[0, 0] = x[0, 0] + z[0] * cos(z[1] + x[2, 0])
    zp[0, 1] = x[1, 0] + z[0] * sin(z[1] + x[2, 0])
    return zp


def get_n_lms(lm):
    return lm.shape[0]
