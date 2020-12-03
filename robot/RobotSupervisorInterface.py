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


# a class representing the available interactions a supervisor may have with a robot
from numpy.random import normal
class RobotSupervisorInterface:

    def __init__(self, robot, robot_cfg):
        """
        Initializes a RobotSupervisorInterface object
        :param robot: The underlying robot
        """
        self.robot = robot
        self.sensor_noise = robot_cfg['sensor']['noise']
        self.motor_noise = (robot_cfg['motor']['noise']['left_velocity'],
                                robot_cfg['motor']['noise']['right_velocity'])

    def read_proximity_sensors(self):
        """
        :return: List of the current proximity sensor readings
        """
        return [s.read() + normal(0, self.sensor_noise) for s in self.robot.ir_sensors]

    def read_wheel_encoders(self):
        """
        :return: List of current wheel encoder readings
        """

        return [e.read() for e in self.robot.wheel_encoders]

    def read_feature_detector(self):
        """
        :return: List of landmark ids
        """
        feature_detector = self.robot.feature_detector.read()
        return [feature_detector[sensor.id] for sensor in self.robot.ir_sensors]

    def set_wheel_drive_rates(self, v_l, v_r):
        """
        Specify the wheel velocities
        :param v_l: Velocity of left wheel
        :param v_r: Velocity of right wheel
        """
        v_l += normal(0, self.motor_noise[0])
        v_r += normal(0, self.motor_noise[1])
        self.robot.set_wheel_drive_rates(v_l, v_r)
