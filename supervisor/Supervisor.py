# Sobot Rimulator - A Robot Programming Tool
# Copyright (C) 2013-2014 Nicholas S. D. McCrea
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Email mccrea.engineering@gmail.com for questions, comments, or to report bugs.
from supervisor.controllers.FastSlam import FastSlam
from supervisor.controllers.GTGAndAOController import *
from supervisor.controllers.FollowWallController import *
from supervisor.controllers.GoToAngleController import *
from supervisor.controllers.EKFSlam import *
from supervisor.SupervisorControllerInterface import *
from supervisor.SupervisorStateMachine import *


class Supervisor:

    def __init__(self, robot_interface,  # the interface through which this supervisor will interact with the robot
                 cfg,
                 goal=[0.0, 0.0],  # the goal to which this supervisor will guide the robot
                 initial_pose_args=[0.0, 0.0, 0.0]):  # the pose the robot will have when control begins

        # Extract relevant configs
        self.robot_cfg = cfg["robot"]
        self.control_cfg = cfg["control"]

        # internal clock time in seconds
        self.time = 0.0

        # robot representation
        # NOTE: the supervisor does NOT have access to the physical robot, only the robot's interface
        self.robot = robot_interface

        # proximity sensor information
        self.proximity_sensor_placements = [Pose(rawpose[0], rawpose[1], radians(rawpose[2])) for rawpose in
                                            self.robot_cfg["sensor"]["poses"]]
        self.proximity_sensor_min_range = self.robot_cfg["sensor"]["min_range"]
        self.proximity_sensor_max_range = self.robot_cfg["sensor"]["max_range"]
        self.proximity_sensor_min_read_value = self.robot_cfg["sensor"]["min_read_value"]
        self.proximity_sensor_max_read_value = self.robot_cfg["sensor"]["max_read_value"]
        self.sensor_conversion_factor = log(
            self.proximity_sensor_min_read_value / self.proximity_sensor_max_read_value) / (
                self.proximity_sensor_max_range - self.proximity_sensor_min_range)

        # odometry information
        self.robot_wheel_radius = self.robot_cfg["wheel"]["radius"]
        self.robot_wheel_base_length = self.robot_cfg["wheel"]["base_length"]
        self.wheel_encoder_ticks_per_revolution = self.robot_cfg["wheel"]["ticks_per_rev"]
        self.prev_ticks_left = 0
        self.prev_ticks_right = 0

        # controllers
        controller_interface = SupervisorControllerInterface(self)
        self.go_to_angle_controller = GoToAngleController(controller_interface)
        self.go_to_goal_controller = GoToGoalController(controller_interface)
        self.avoid_obstacles_controller = AvoidObstaclesController(controller_interface)
        self.gtg_and_ao_controller = GTGAndAOController(controller_interface)
        self.follow_wall_controller = FollowWallController(controller_interface)

        # slam
        self.slam = EKFSlam(controller_interface, cfg["map"]["obstacle"]["radius"], cfg["slam"])
        # self.slam = FastSlam(controller_interface, step_time=1/20)

        # state machine
        self.state_machine = SupervisorStateMachine(self, self.control_cfg)

        # state
        self.proximity_sensor_distances = [0.0, 0.0] * len(self.robot_cfg["sensor"]["poses"])  # sensor distances
        self.estimated_pose = Pose(*initial_pose_args)  # estimated pose
        self.current_controller = self.go_to_goal_controller  # current controller

        # goal
        self.goal = goal

        # control bounds
        self.v_max = self.robot_cfg["max_transl_vel"]
        self.omega_max = self.robot_cfg["max_ang_vel"]

        # CONTROL OUTPUTS - UNICYCLE
        self.v_output = 0.0
        self.omega_output = 0.0

        self.v_l = 0.0
        self.v_r = 0.0

    # simulate this supervisor running for one time increment
    def step(self, dt):
        # increment the internal clock time
        self.time += dt

        # NOTE: for simplicity, we assume that the onboard computer executes exactly one control loop for every simulation time increment
        # although technically this is not likely to be realistic, it is a good simplificiation

        # execute one full control loop
        self._execute(dt)

    # execute one control loop
    def _execute(self, dt):
        self._update_state()  # update state
        self.current_controller.execute()  # apply the current controller
        v, yaw = self._diff_to_uni(self.v_l, self.v_r)
        self.slam.ekf_slam(np.array([[v], [yaw]]), self.proximity_sensor_distances, dt)
        # self.slam.fast_slam(np.array([[v], [yaw]]), self.proximity_sensor_distances)
        self._send_robot_commands()  # output the generated control signals to the robot

    # update the estimated robot state and the control state
    def _update_state(self):
        # update estimated robot state from sensor readings
        self._update_proximity_sensor_distances()
        self._update_odometry()

        # calculate new heading vectors for each controller
        self._update_controller_headings()

        # update the control state
        self.state_machine.update_state()

    # calculate updated heading vectors for the active controllers
    def _update_controller_headings(self):
        self.go_to_goal_controller.update_heading()
        self.avoid_obstacles_controller.update_heading()
        self.gtg_and_ao_controller.update_heading()
        self.follow_wall_controller.update_heading()

    # update the distances indicated by the proximity sensors
    def _update_proximity_sensor_distances(self):
        self.proximity_sensor_distances = \
            [self.proximity_sensor_min_range + (
                log(readval / self.proximity_sensor_max_read_value)) / self.sensor_conversion_factor
             for readval in self.robot.read_proximity_sensors()]

    # update the estimated position of the robot using it's wheel encoder readings
    def _update_odometry(self):
        R = self.robot_wheel_radius
        N = float(self.wheel_encoder_ticks_per_revolution)

        # read the wheel encoder values
        ticks_left, ticks_right = self.robot.read_wheel_encoders()

        # get the difference in ticks since the last iteration
        d_ticks_left = ticks_left - self.prev_ticks_left
        d_ticks_right = ticks_right - self.prev_ticks_right

        # estimate the wheel movements
        d_left_wheel = 2 * pi * R * (d_ticks_left / N)
        d_right_wheel = 2 * pi * R * (d_ticks_right / N)
        d_center = 0.5 * (d_left_wheel + d_right_wheel)

        # calculate the new pose
        prev_x, prev_y, prev_theta = self.estimated_pose.sunpack()
        new_x = prev_x + (d_center * cos(prev_theta))
        new_y = prev_y + (d_center * sin(prev_theta))
        new_theta = prev_theta + ((d_right_wheel - d_left_wheel) / self.robot_wheel_base_length)

        # update the pose estimate with the new values
        self.estimated_pose.supdate(new_x, new_y, new_theta)

        # save the current tick count for the next iteration
        self.prev_ticks_left = ticks_left
        self.prev_ticks_right = ticks_right

    # generate and send the correct commands to the robot
    def _send_robot_commands(self):
        # limit the speeds:
        v = max(min(self.v_output, self.v_max), -self.v_max)
        omega = max(min(self.omega_output, self.omega_max), -self.omega_max)

        # send the drive commands to the robot
        v_l, v_r = self._uni_to_diff(v, omega)
        self.v_l = v_l
        self.v_r = v_r
        self.robot.set_wheel_drive_rates(v_l, v_r)

    def _uni_to_diff(self, v, omega):
        # v = translational velocity (m/s)
        # omega = angular velocity (rad/s)

        R = self.robot_wheel_radius
        L = self.robot_wheel_base_length

        v_l = ((2.0 * v) - (omega * L)) / (2.0 * R)
        v_r = ((2.0 * v) + (omega * L)) / (2.0 * R)

        return v_l, v_r

    def _diff_to_uni(self, v_l, v_r):
        # v_l = left-wheel angular velocity (rad/s)
        # v_r = right-wheel angular velocity (rad/s)

        R = self.robot_wheel_radius
        L = self.robot_wheel_base_length

        v = (R / 2.0) * (v_r + v_l)
        omega = (R / L) * (v_r - v_l)

        return v, omega
