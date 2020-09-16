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


# an interfacing allowing a controller to interact with its supervisor

class SupervisorControllerInterface:

    def __init__(self, supervisor):
        """
        Initializes a SupervisorControllerInterface object
        :param supervisor: The underlying supervisor
        """
        self.supervisor = supervisor

    def current_state(self):
        """
        :return: The current control state
        """
        return self.supervisor.state_machine.current_state

    def estimated_pose(self):
        """
        :return: The supervisors internal pose estimation based on odometry
        """
        return self.supervisor.estimated_pose

    def estimated_pose_ekfslam(self):
        """
        :return: The pose that is currently estimated by the EKF SLAM
        """
        if self.supervisor.ekfslam is not None:
            return self.supervisor.ekfslam.get_estimated_pose()
        else:
            None

    def estimated_pose_fastslam(self):
        """
        :return: The pose that is currently estimated by the FastSLAM
        """
        if self.supervisor.fastslam is not None:
            return self.supervisor.fastslam.get_estimated_pose()
        else:
            None

    def proximity_sensor_placements(self):
        """
        :return: The placement poses of the robot's sensors
        """
        return self.supervisor.proximity_sensor_placements

    def proximity_sensor_distances(self):
        """
        :return: The robots proximity sensor read values converted to real distances in meters
        """
        return self.supervisor.proximity_sensor_distances

    def read_wheel_encoders(self):
        """
        Read the wheel encoder values and return the parameters of robot
        """
        params = {
            "wheel_base_length": self.supervisor.robot_wheel_base_length,
            "wheel_radius": self.supervisor.robot_wheel_radius,
            "wheel_encoder_ticks_per_revolution": self.supervisor.wheel_encoder_ticks_per_revolution
        }
        ticks_left, ticks_right = self.supervisor.robot.read_wheel_encoders()
        return (ticks_left, ticks_right), params


    def proximity_sensor_distances_from_robot_center(self):
        """
        :return: The robots proximity sensor read values converted to real distances in meters
        """
        return self.supervisor.proximity_sensor_distances_from_robot_center

    def proximity_sensor_max_range(self):
        return self.supervisor.proximity_sensor_max_range

    def proximity_sensor_min_range(self):
        """
        :return: The maximum sensor range of the robots proximity sensors
        """
        return self.supervisor.proximity_sensor_min_range

    def proximity_sensor_positive_detections(self):
        """
        :return: List of boolean values indicating which sensors are actually detecting obstacles
        """
        sensor_range = self.supervisor.proximity_sensor_max_range
        return [d < sensor_range - 0.001 for d in self.proximity_sensor_distances()]

    def v_max(self):
        """
        :return: The maximum velocity of the supervisor
        """
        return self.supervisor.v_max

    def goal(self):
        """
        :return: The supervisors goal
        """
        return self.supervisor.goal

    def time(self):
        """
        :return: The supervisors internal clock time
        """
        return self.supervisor.time

    def set_outputs(self, v, omega):
        """
        Specify the next motion command
        :param v: Translational veloctiy (m/s)
        :param omega: otational velocity (rad/s)
        """
        self.supervisor.v_output = v
        self.supervisor.omega_output = omega
