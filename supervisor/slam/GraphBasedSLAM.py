from supervisor.slam.graph.landmark_graph import *
from supervisor.slam.Slam import Slam
from utils.math_util import normalize_angle
import numpy as np
from math import *
from models.Pose import Pose
from time import sleep

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
        # self.sensor_noise = np.diag([slam_cfg["graph_based_slam"]["sensor_noise"]["x"],
        #                              slam_cfg["graph_based_slam"]["sensor_noise"]["y"]])**2
        # self.motion_noise = np.diag([slam_cfg["graph_based_slam"]["motion_noise"]["x"],
        #                              slam_cfg["graph_based_slam"]["motion_noise"]["y"],
        #                              np.deg2rad(slam_cfg["graph_based_slam"]["motion_noise"]["theta"])]) ** 2
        # self.min_distance_threshold = slam_cfg["graph_based_slam"]["distance_threshold"]
        """              DEBUG             """
        self.sensor_noise = np.diag([0.1, 0.1])**2
        self.motion_noise = np.diag([0.1, 0.1, np.deg2rad(30)]) ** 2
        self.min_distance_threshold = 0.1
        """                                 """

        # The estimated combined state vector, initially containing the robot pose at the origin and no landmarks
        self.mu = np.zeros((self.robot_state_size, 1))
        # The state covariance, initially set to absolute certainty of the initial robot pose
        self.Sigma = np.zeros((self.robot_state_size, self.robot_state_size))
        self.step_counter = 0

        self.max_range = self.supervisor.proximity_sensor_max_range()
        self.min_range = self.supervisor.proximity_sensor_min_range()
        # self.robot_width = self.supervisor.wheel_base_length()
        # print (self.max_range, self.min_range)

        self.graph = LMGraph()
        self.old_wheel_record = (0, 0)
        self.odom_pose = self.__reset_odometry_measurement() # accumulative odometry estimation from wheel encoders

        self.step_counter = 0
        self.fix_hessian = 0
        self.__init_first_step()


    def __init_first_step(self):
        """    add first pose-vertex     """
        # z = self.interface.read_proximity_sensors()  # raw measurements from sensor
        z = []
        vertex1 = PoseVertex(self.mu, np.eye(3), z)
        self.graph.add_vertex(vertex1)
        self.fix_hessian += 3
        """    add landmarks relative the first pose vertex     """

        # for zi in z:
        #     pos_lm = self.calc_landmark_position(self.mu, zi)
        #     vertex2 = LandmarkVertex(pos_lm, self.Q)
        #     self.graph.add_vertex(vertex2)
        #     meas, info = self.__convert_pose_landmark_raw_measurement(zi)
        #     self.graph.add_edge(vertex1, vertex2, meas, info)
        #     self.fix_hessian += 2


    def get_estimated_pose(self):
        """
        Returns the estimated pose of the robot
        """
        return Pose(self.mu[0, 0], self.mu[1, 0], self.mu[2, 0])
        #return Pose(self.odom_pose[0, 0], self.odom_pose[1, 0], self.odom_pose[2, 0])

    def get_landmarks(self):
        """
        Returns the estimated landmark positions
        """
        landmarks = []
        vertices = self.graph.get_estimated_landmark_vertices()
        for v in vertices:
            landmarks.append((v.pose[0], v.pose[1]))
        return landmarks

    def __update_odometry_estimation(self, odom_pose):
        """
        Calculate odometry estimation from wheel encoders.
        """
        ticks, params = self.supervisor.read_wheel_encoders()  #  (ticks_left, ticks_right)
        base_length = params['wheel_base_length']
        R = params['wheel_radius']
        N = params['wheel_encoder_ticks_per_revolution']
        left_wheel_movement = 2 * np.pi * R * ticks[0] / N
        right_wheel_movement = 2 * np.pi * R * ticks[1] / N
        wheel_record = (left_wheel_movement, right_wheel_movement)
        vl = (wheel_record[0] - self.old_wheel_record[0])/self.dt  # Velocity of the left wheel
        vr = (wheel_record[1] - self.old_wheel_record[1])/self.dt  # Velocity of the right wheel
        v, w = self.diff_to_uni(vl, vr, base_length)   # Convert to uni-wheel model
        #print ("v, w", v, w)
        # Update the odometry estimation
        odom_pose = self.motion_model(odom_pose, np.array([[v], [w]]), self.dt)
        self.old_wheel_record = np.copy(wheel_record)
        return odom_pose

    def __reset_odometry_measurement(self):
        return np.copy(self.mu)

    def update(self, u, z):
        """
        Executes an update cycle of the SLAM algorithm
        :param u: Motion command, A single item is a vector of [[translational_velocity], [angular_velocity]]
        :param z: List of measurements. A single item is a tuple of (range, angle)
        """

        self.mu = self.motion_model(self.mu, u, self.dt) # Calculate next step
        J = self.jaco_motion_model(self.mu, u, self.dt)  # Calculate jacobian matrix
        # print ("helo", self.Sigma.shape, J.shape, self.motion_noise.shape)
        self.Sigma = J.T @ self.Sigma @ J + self.motion_noise  # Update covariance matrix
        self.odom_pose = self.__update_odometry_estimation(self.odom_pose) # Update accumulative odometry estimation
        self.step_counter += 1

        """        Update the Graph         """
        if self.step_counter >  5: # create a vertex each n steps
            print('[COUNT] POSEES, LANDMARKS :', self.graph.count_vertices())
            self.__front_end(z)
            self.step_counter = 0
            self.odom_pose = self.__reset_odometry_measurement() # reset odom_pose
            num_poses, _ = self.graph.count_vertices()

            if num_poses % 5 == 0 and num_poses > 0:
                self.__back_end() # graph optimization
                last_vertex = self.graph.get_last_pose_vertex()
                self.mu = np.copy(last_vertex.pose)   # update current state
                self.Sigma = np.copy(self.motion_noise)
                self.odom_pose = self.__reset_odometry_measurement()  # Reset odometry estimation


            self.graph.draw()


    def __front_end(self, z):
        """

        :param z: list of range-bearing measurement, a single item is (distance, angle) related to robot
        :return:
        """
        """    calculate the next vertex of poses     """
        vertex1 = self.graph.get_last_pose_vertex()
        vertex2 = PoseVertex(self.mu, self.Sigma, z)
        self.graph.add_vertex(vertex2)

        """     calculate landmark vertices    """
        # for i, zi in enumerate(z):
        #     # Only execute if sensor observed landmark
        #     if not self.supervisor.proximity_sensor_positive_detections()[i]:
        #         print ("No observation", i)
        #         continue
        #     print (np.rad2deg(zi))
        #     pos_lm = self.calc_landmark_position(self.mu, zi) # x-y position in world coordinate
        #     min_index, vertices_lm = self.__data_association(self.mu, self.Sigma, pos_lm)
        #     N = len(vertices_lm)
        #     print ("min_index, N", min_index, N)
        #     if min_index == N:
        #         print ("new landmark was found:", N)
        #         vertex3 = LandmarkVertex(pos_lm, self.sensor_noise)  # create a landmark vertex
        #         self.graph.add_vertex(vertex3)
        #     else:
        #         vertex3 = vertices_lm[min_index]
        #     # calculate actual measurement and information matrix
        #     meas, info = self.__convert_pose_landmark_raw_measurement(zi)
        #     self.graph.add_edge(vertex2, vertex3, meas, info)

        """      calculate pose-pose edge       """
        meas, info = self.__convert_pose_pose_raw_measurement(vertex1.pose, self.odom_pose)
        self.graph.add_edge(vertex1, vertex2, meas, info)

        #
        # plt.cla()
        # plt.plot(vertex1.pose[0,0], vertex1.pose[1,0], 'k*')
        # plt.plot(vertex2.pose[0,0], vertex2.pose[1,0], 'r*')
        # plt.plot(self.odom_pose[0,0], self.odom_pose[1,0], 'gx')
        # plt.show(block=True)


    def __back_end(self):
        print('SLAM OPTIMIZATION...')
        error = self.graph.graph_optimization(number_fix=self.fix_hessian, damp_factor=1)
        print('SLAM OPTIMIZATION: GLOBAL ERROR', error)

    def __convert_pose_landmark_raw_measurement(self, zi):
        info = inv(self.sensor_noise)
        meas_xy = np.array([[zi[0] * cos(zi[1])],
                            [zi[0] * sin(zi[1])]])
        return meas_xy, info

    def __convert_pose_pose_raw_measurement(self, pose, odom_pose):
        """
        Convert raw measurement u to the measurement defined by the error metric
        """
        pose1 = pose
        pose2 = odom_pose  # expected pose of next step
        M = inv(v2t(pose1)) @ v2t(pose2)  # transformation matrix from x_meas to vertex1.pose
        info = inv(self.motion_noise)
        meas = t2v(M)
        return meas, info


    def __calc_range_bearing_delta(self, x, lm, z):
        delta = lm - x[0:2, :]
        range = np.linalg.norm(delta, axis=0)
        phi = np.arctan2(delta[1, :], delta[0, :])
        ext_z = np.vstack((range, phi))
        err = ext_z - np.array(z).reshape((2,1))
        err[1, :] = np.arctan2(np.sin(err[1, :]), np.cos(err[1, :]))
        return err


    def __data_association(self, mu, sigma, zi):
        """

        :param mu: robot pose in current step [x, y, theta].T
        :param sigma: covariance matrix of the robot pose. 3x3 matrix
        :param zi: a measurement of landmark in world coordinate. [x, y].T
        :return:
            min_index: index of the nearest landmark in list vertices_lm
            vertices_lm: index of the vertex of this landmark
        """

        lms, vertices_lm = self.get_estimated_landmark_position() # find all landmark vertices from the list
        N = len(lms) # number of known landmarks
        if N == 0:  # there were no landmarks being found
            return N, []
        else:
            print("__data_association", N)
            lms = np.array(lms).T # xy position of landmarks in world coordinate
            distances = np.linalg.norm(lms - zi, axis=0)
            distances = np.append(distances, self.min_distance_threshold)
            min_index = np.argmin(distances)
            print (distances, "__data_association")

            # zi is a measurement of a new landmark, if min_index == N
            # zi is a measurement of an old landmark, if min_index < N
            return min_index, vertices_lm

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
        """
        :param v_l: Translational velocity of the left wheel
        :param v_r: Translational velocity of the right wheel
        :return v, w
                Translational and angular velocities
        """
        v = (v_r + v_l) * 0.5
        w = (v_r - v_l) / width
        return v, w