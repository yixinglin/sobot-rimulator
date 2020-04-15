"""
FastSLAM 1.0
Based on implementation of Atsushi Sakai (https://github.com/AtsushiSakai/PythonRobotics)
Most significant changes made:
- Add support for a flexible number of landmarks
- Add support for unknown data association
- Change ekf update
- Change resampling algorithm
"""

from math import cos, sin, sqrt, atan2, exp, pi

import numpy as np

# Fast SLAM covariance
from models.Pose import Pose
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle

sensor_noise = np.diag([0.2, np.deg2rad(30.0)]) ** 2
motion_noise = np.diag([0.005, 0.005]) ** 2


class Particle:

    def __init__(self, lm_state_size):
        """
        A particle is initialized at the origin position with no observed landmarks and an importance factor of 1
        :param lm_state_size: The state size for a landmark
        """
        self.w = 1.0
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        # landmark x-y positions
        self.lm = np.zeros((0, lm_state_size))
        # landmark position covariance
        self.lmP = np.zeros((0, lm_state_size))


class FastSlam(Slam):

    def __init__(self, supervisor_interface, slam_cfg, step_time):
        """
        Creates a FastSlam object
        :param supervisor_interface: The interface to interact with the robot supervisor
        :param slam_cfg: The configuration for the SLAM algorithm
        :param step_time: The discrete time that a single simulation cycle increments
        """
        self.supervisor = supervisor_interface
        self.dt = step_time
        self.distance_threshold = slam_cfg["fast_slam"]["distance_threshold"]
        self.n_particles = slam_cfg["fast_slam"]["n_particles"]
        self.robot_state_size = slam_cfg["robot_state_size"]
        self.landmark_state_size = slam_cfg["landmark_state_size"]
        self.particles = [Particle(self.landmark_state_size) for _ in range(self.n_particles)]

    def get_estimated_pose(self):
        """
        Returns the estimated robot pose by only considering the particle with the highest importance factor
        :return: Estimated robot pose consisting of position and angle
        """
        particle = self.get_best_particle()
        return Pose(particle.x, particle.y, particle.theta)

    def get_landmarks(self):
        """
        Returns the estimated landmark positions by only considering the particle with the highest importance factor
        :return: List of estimated landmark positions
        """
        particle = self.get_best_particle()
        return [(x, y) for (x, y) in zip(particle.lm[:, 0], particle.lm[:, 1])]

    def execute(self, u, z):
        """
        Performs a full update step of the FastSLAM algorithm
        :param u: Motion command
        :param z: Sensor measurements
        :return: Updated list of particles
        """
        # prediction step
        self.particles = self.predict_particles(self.particles, u)
        # correction step
        self.correction_step(z)
        return self.particles

    def correction_step(self, z):
        """
        Performs the correction step
        :param z: Measurement
        """
        self.particles = self.measurement_update(self.particles, z)
        self.particles = self.resampling(self.particles)

    def predict_particles(self, particles, u):
        """
        Performs the prediction step of the algorithm
        :param particles: List of particles
        :param u: Motion command
        :return: List of predicted particles after applying motion command
        """
        for particle in particles:
            px = np.zeros((self.robot_state_size, 1))
            # Copy the current particle
            px[0, 0] = particle.x
            px[1, 0] = particle.y
            px[2, 0] = particle.theta
            # Apply noise to the motion command
            u += (np.random.randn(1, 2) @ motion_noise ** 0.5).T
            # Apply noise-free motion with noisy motion command
            px = self.motion_model(px, u, self.dt)
            # Update particle
            particle.x = px[0, 0]
            particle.y = px[1, 0]
            particle.theta = px[2, 0]
        return particles

    def measurement_update(self, particles, z):
        """
        Performs the measurement update of the algorithm, which consists of data association
        1. data association
        2. adding a new landmark or
           computing importance factor and performing an EKF update for an already encountered landmark
        :param particles:
        :param z:
        :return:
        """
        # Removing the importance factors of the previous cycle
        particles = self.clear_importance_factors(particles)
        for i, (distance, theta) in enumerate(z):
            measurement = np.asarray([distance, theta])
            # Skip the measuremnt if no landmark was detected
            if not self.supervisor.proximity_sensor_positive_detections()[i]:
                continue
            for particle in particles:
                lm_id = self.data_association(particle, measurement)
                nLM = get_n_lms(particle.lm)
                if lm_id == nLM:  # If the landmark is new
                    self.add_new_lm(particle, measurement)
                else:
                    self.update_landmark(particle, measurement, lm_id)

        return particles

    def data_association(self, particle, z):
        """
        Associates the measurement to a landmark.
        Chooses the closest landmark to the measured location
        :param particle: Particle that will be updated
        :param z: Measurement
        :return: The id of the landmark that is associated to the measurement
        """
        nLM = get_n_lms(particle.lm)
        mdist = []
        # Calculate measured landmark position
        measured_lm = calc_landmark_position(particle, z)
        # Calculate distance from measured landmark position to all other landmark positions
        for i in range(nLM):
            lm_i = particle.lm[i]
            delta = lm_i - measured_lm
            distance = sqrt(delta[0, 0] ** 2 + delta[0, 1] ** 2)
            mdist.append(distance)
        # Use distance threshold as criterium for spotting new landmark
        mdist.append(self.distance_threshold)
        # Choose the landmark that is closest to the measured location
        min_id = mdist.index(min(mdist))
        return min_id

    def normalize_weight(self, particles):
        """
        Normalizes the importance factors of the particles so that their sum is 1
        Special case: If sum is 0, then all particles receive the same importance factor
        :param particles: List of particles
        :return: List of particles with normalized importance factors
        """
        sumw = sum([p.w for p in particles])
        try:
            for particle in particles:
                particle.w /= sumw
        except ZeroDivisionError:
            for particle in particles:
                particle.w = 1.0 / self.n_particles
        return particles

    def clear_importance_factors(self, particles):
        """
        Sets all importance factors to the same value
        :param particles: List of particles
        :return: List of particles with same importance factors
        """
        for particle in particles:
            particle.w = 1.0 / self.n_particles
        return particles

    def get_best_particle(self):
        """
        Returns the particle with the highest importance factor
        :return: Particle with highest importance factor
        """
        get_weight = lambda particle: particle.w
        return max(self.particles, key=get_weight)

    def add_new_lm(self, particle, z):
        """
        Initializes a yet unknown landmark using measurement
        :param particle: Particle that will be updated
        :param z: Measurement
        :return: Particle with a new landmark location and landmark uncertainty added
        """
        r = z[0]
        b = z[1]

        measured_x = cos(normalize_angle(particle.theta + b))
        measured_y = sin(normalize_angle(particle.theta + b))
        # Calculate landmark location
        new_lm = np.array([particle.x + r * measured_x, particle.y + r * measured_y]).reshape(1, self.landmark_state_size)
        particle.lm = np.vstack((particle.lm, new_lm))

        # Calculate initial covariance
        Gz = np.array([[measured_x, -r * measured_y],
                       [measured_y, r * measured_x]])
        particle.lmP = np.vstack((particle.lmP, Gz @ sensor_noise @ Gz.T))

        return particle

    def update_landmark(self, particle, z, lm_id):
        """
        Updates the estimated landmark position and uncertainties as well as the particles importance factor
        :param particle: Particle that is being updated
        :param z: Measurement
        :param lm_id: Id of the landmark that is associated to the measurement
        :return: Updated particle
        """
        landmark = np.array(particle.lm[lm_id, :]).reshape(2, 1)
        landmark_cov = np.array(particle.lmP[2 * lm_id:2 * lm_id + 2, :])

        # Computing difference between landmark and robot position
        delta_x = landmark[0, 0] - particle.x
        delta_y = landmark[1, 0] - particle.y
        # Computing squared distance
        q = delta_x ** 2 + delta_y ** 2
        sq = sqrt(q)
        # Computing the measurement that would be expected
        innovation = np.array(
            [sq, normalize_angle(atan2(delta_y, delta_x) - particle.theta)]).reshape(2, 1)
        # Computing the Jacobian
        H = np.array([[delta_x / sq, delta_y / sq],
                      [-delta_y / q, delta_x / q]])
        # Computing the covariance of the measurement
        Psi = H @ landmark_cov @ H.T + sensor_noise
        # Computing difference between actual measurement and expected measurement
        dz = z.reshape(2, 1) - innovation
        dz[1, 0] = normalize_angle(dz[1, 0])

        landmark, landmark_cov = self.ekf_update(landmark, landmark_cov, dz, H, Psi)
        particle.lm[lm_id, :] = landmark.T
        particle.lmP[2 * lm_id:2 * lm_id + 2, :] = landmark_cov
        # Multiplying importance factors, since this is just the weight for a single sensor measurement
        particle.w *= self.compute_importance_factor(dz, Psi)

        return particle

    def compute_importance_factor(self, dz, Psi):
        """
        Computes an importance factor.
        :param dz: Difference between actual measurement and innovation (expected measurement)
        :param Psi: Covariance matrix for measurement
        :return: Importance factor
        """
        num = exp(-0.5 * dz.T @ np.linalg.inv(Psi) @ dz)
        den = sqrt(2.0 * pi * np.linalg.det(Psi))
        w = num / den
        return w

    def ekf_update(self, landmark, landmark_cov, dz, H, Psi):
        """
        Updates the landmark position and covariance
        :param landmark: Estimated landmark position
        :param landmark_cov: Landmark covariance
        :param dz: Difference between actual measurement and innovation (expected measurement)
        :param H: Jacobian of the measurement
        :param Psi: Covariance of the measurement
        :return: updated estimated landmark position, updated landmark covariance
        """
        K = (landmark_cov @ H.T) @ np.linalg.inv(Psi)
        landmark += K @ dz
        landmark_cov = (np.identity(len(landmark_cov)) - (K @ H)) @ landmark_cov
        return landmark, landmark_cov

    def resampling(self, particles):
        """
        Resamples the particles based on their importance factors.
        :param particles: list Particles with importance factors
        :return: List of particles resampled based on their importance factors
        """
        particles = self.normalize_weight(particles)
        weights = np.array([particle.w for particle in particles])
        wcum = np.cumsum(weights)
        unif = np.random.rand(self.n_particles)
        inds = []
        for random in unif:
            i = 0
            while random > wcum[i]:
                i += 1
            inds.append(i)
        tparticles = particles
        for i in range(len(inds)):
            particles[i].x = tparticles[inds[i]].x
            particles[i].y = tparticles[inds[i]].y
            particles[i].theta = tparticles[inds[i]].theta
            particles[i].lm = tparticles[inds[i]].lm[:, :]
            particles[i].lmP = tparticles[inds[i]].lmP[:, :]
            particles[i].w = tparticles[inds[i]].w
        return particles

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
        res[2] = normalize_angle(res[2])
        return res


def calc_landmark_position(particle, z):
    zp = np.zeros((1, 2))
    zp[0, 0] = particle.x + z[0] * cos(z[1] + particle.theta)
    zp[0, 1] = particle.y + z[0] * sin(z[1] + particle.theta)
    return zp


def get_n_lms(lm):
    return lm.shape[0]
