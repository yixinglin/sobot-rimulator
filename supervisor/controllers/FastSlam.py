"""
FastSLAM 1.0 example
author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np

# Fast SLAM covariance
from models.Pose import Pose

Q = np.diag([3.0, np.deg2rad(10.0)]) ** 2
R = np.diag([1.0, np.deg2rad(20.0)]) ** 2

M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM srate size [x,y]
N_PARTICLE = 100  # number of particle
NTH = N_PARTICLE / 1.5  # Number of particle for re-sampling


class Particle:

    def __init__(self, N_LM):
        self.w = 1.0 / N_PARTICLE
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        # landmark x-y positions
        self.lm = np.zeros((N_LM, LM_SIZE))
        # landmark position covariance
        self.lmP = np.zeros((N_LM * LM_SIZE, LM_SIZE))

# Technically FastSlam1


class FastSlam:

    def __init__(self, supervisor_interface, step_time):
        self.supervisor = supervisor_interface
        self.dt = step_time
        self.particles = [Particle(20) for _ in range(N_PARTICLE)]

    def fast_slam(self, u, z):
        self.particles = self.predict_particles(self.particles, u)

        self.particles = self.update_with_observation(self.particles, z)

        self.particles = self.resampling(self.particles)

        return self.particles

    def get_estimated_pose(self):
        xEst = self.calc_final_state(self.particles)
        return Pose(xEst[0, 0], xEst[1, 0], xEst[2, 0])

    def get_landmarks(self):
        xEst = self.calc_final_state(self.particles)
        return [(x, y) for (x, y) in zip(xEst[STATE_SIZE::2], xEst[STATE_SIZE+1::2])]

    def normalize_weight(self, particles):
        sumw = sum([p.w for p in particles])

        try:
            for i in range(N_PARTICLE):
                particles[i].w /= sumw
        except ZeroDivisionError:
            for i in range(N_PARTICLE):
                particles[i].w = 1.0 / N_PARTICLE

            return particles

        return particles

    def calc_final_state(self, particles):
        xEst = np.zeros((STATE_SIZE, 1))

        particles = self.normalize_weight(particles)

        for i in range(N_PARTICLE):
            xEst[0, 0] += particles[i].w * particles[i].x
            xEst[1, 0] += particles[i].w * particles[i].y
            xEst[2, 0] += particles[i].w * particles[i].yaw

        xEst[2, 0] = self.pi_2_pi(xEst[2, 0])
        #  print(xEst)

        return xEst

    def predict_particles(self, particles, u):
        for i in range(N_PARTICLE):
            px = np.zeros((STATE_SIZE, 1))
            px[0, 0] = particles[i].x
            px[1, 0] = particles[i].y
            px[2, 0] = particles[i].yaw
            ud = u + (np.random.randn(1, 2) @ R ** 0.5).T  # add noise
            px = self.motion_model(px, ud)
            particles[i].x = px[0, 0]
            particles[i].y = px[1, 0]
            particles[i].yaw = px[2, 0]

        return particles

    def add_new_lm(self, particle, z, Q_cov):
        r = z[0]
        b = z[1]
        lm_id = int(z[2])

        s = math.sin(self.pi_2_pi(particle.yaw + b))
        c = math.cos(self.pi_2_pi(particle.yaw + b))

        particle.lm[lm_id, 0] = particle.x + r * c
        particle.lm[lm_id, 1] = particle.y + r * s

        # covariance
        Gz = np.array([[c, -r * s],
                       [s, r * c]])

        particle.lmP[2 * lm_id:2 * lm_id + 2] = Gz @ Q_cov @ Gz.T

        return particle

    def compute_jacobians(self, particle, xf, Pf, Q_cov):
        dx = xf[0, 0] - particle.x
        dy = xf[1, 0] - particle.y
        d2 = dx ** 2 + dy ** 2
        d = math.sqrt(d2)

        zp = np.array(
            [d, self.pi_2_pi(math.atan2(dy, dx) - particle.yaw)]).reshape(2, 1)

        Hv = np.array([[-dx / d, -dy / d, 0.0],
                       [dy / d2, -dx / d2, -1.0]])

        Hf = np.array([[dx / d, dy / d],
                       [-dy / d2, dx / d2]])

        Sf = Hf @ Pf @ Hf.T + Q_cov

        return zp, Hv, Hf, Sf

    def update_kf_with_cholesky(self, xf, Pf, v, Q_cov, Hf):
        PHt = Pf @ Hf.T
        S = Hf @ PHt + Q_cov

        S = (S + S.T) * 0.5
        SChol = np.linalg.cholesky(S).T
        SCholInv = np.linalg.inv(SChol)
        W1 = PHt @ SCholInv
        W = W1 @ SCholInv.T

        x = xf + W @ v
        P = Pf - W1 @ W1.T

        return x, P

    def update_landmark(self, particle, z, Q_cov):
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q)

        dz = z[0:2].reshape(2, 1) - zp
        dz[1, 0] = self.pi_2_pi(dz[1, 0])

        xf, Pf = self.update_kf_with_cholesky(xf, Pf, dz, Q_cov, Hf)

        particle.lm[lm_id, :] = xf.T
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = Pf

        return particle

    def compute_weight(self, particle, z, Q_cov):
        lm_id = int(z[2])
        xf = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        Pf = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2])
        zp, Hv, Hf, Sf = self.compute_jacobians(particle, xf, Pf, Q_cov)

        dx = z[0:2].reshape(2, 1) - zp
        dx[1, 0] = self.pi_2_pi(dx[1, 0])

        try:
            invS = np.linalg.inv(Sf)
        except np.linalg.linalg.LinAlgError:
            print("singuler")
            return 1.0

        num = math.exp(-0.5 * dx.T @ invS @ dx)
        den = 2.0 * math.pi * math.sqrt(np.linalg.det(Sf))

        w = num / den

        return w

    def update_with_observation(self, particles, z):
        z = zip(z, [pose.theta for pose in self.supervisor.proximity_sensor_placements()], range(len(z)))
        for iz, (distance, theta, lmid) in enumerate(z):
            if distance >= self.supervisor.proximity_sensor_max_range() - 0.01:  # only execute if landmark is observed
                continue

            for ip in range(N_PARTICLE):
                # new landmark
                if abs(particles[ip].lm[lmid, 0]) <= 0.01:
                    particles[ip] = self.add_new_lm(particles[ip], np.asarray([distance, theta, lmid]), Q)
                # known landmark
                else:
                    w = self.compute_weight(particles[ip], np.asarray([distance, theta, lmid]), Q)
                    particles[ip].w *= w
                    particles[ip] = self.update_landmark(particles[ip], np.asarray([distance, theta, lmid]), Q)

        return particles

    def resampling(self, particles):
        """
        low variance re-sampling
        """

        particles = self.normalize_weight(particles)

        pw = []
        for i in range(N_PARTICLE):
            pw.append(particles[i].w)

        pw = np.array(pw)

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

    def motion_model(self, x, u):
        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                      [self.dt * math.sin(x[2, 0]), 0],
                      [0.0, self.dt]])

        x = x + B @ u

        x[2, 0] = self.pi_2_pi(x[2, 0])

        return x

    def pi_2_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi
