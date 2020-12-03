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


from robot.DifferentialDriveDynamics import *
from models.Polygon import *
from robot.sensor.ProximitySensor import *
from supervisor.Supervisor import *
from robot.sensor.WheelEncoder import *
from robot.sensor.FeatureDetector import *

class Robot:  # Khepera III robot

    def __init__(self, robot_cfg):
        """
        Initializes a Robot object
        :param robot_cfg: The robot configuration
        """
        self.robot_cfg = robot_cfg
        # geometry
        self.geometry = Polygon(robot_cfg["bottom_plate"])
        self.global_geometry = Polygon(robot_cfg["bottom_plate"])  # actual geometry in world space

        # wheel arrangement
        self.wheel_radius = robot_cfg["wheel"]["radius"]
        self.wheel_base_length = robot_cfg["wheel"]["base_length"]

        # pose
        self.pose = Pose(0.0, 0.0, 0.0)

        # wheel encoders
        self.left_wheel_encoder = WheelEncoder(robot_cfg["wheel"]["ticks_per_rev"])
        self.right_wheel_encoder = WheelEncoder(robot_cfg["wheel"]["ticks_per_rev"])
        self.wheel_encoders = [self.left_wheel_encoder, self.right_wheel_encoder]

        # IR sensors
        self.ir_sensors = []
        for id, _pose in enumerate(robot_cfg["sensor"]["poses"]):
            ir_pose = Pose(_pose[0], _pose[1], radians(_pose[2]))
            self.ir_sensors.append(
                ProximitySensor(self, ir_pose, robot_cfg["sensor"], id))
        # Feature detector
        self.feature_detector = FeatureDetector()

        # dynamics
        self.dynamics = DifferentialDriveDynamics(self.wheel_radius, self.wheel_base_length)

        ## initialize state
        # set wheel drive rates (rad/s)
        self.left_wheel_drive_rate = 0.0
        self.right_wheel_drive_rate = 0.0

    def step_motion(self, dt):
        """
        Simulate the robot's motion over the given time interval
        :param dt: The time interval for which this motion is executed
        """
        v_l = self.left_wheel_drive_rate
        v_r = self.right_wheel_drive_rate

        # apply the robot dynamics to moving parts
        self.dynamics.apply_dynamics(v_l, v_r, dt,
                                     self.pose, self.wheel_encoders)

        # update global geometry
        self.global_geometry = self.geometry.get_transformation_to_pose(self.pose)

        # update all of the sensors
        for ir_sensor in self.ir_sensors:
            ir_sensor.update_position()

    def set_wheel_drive_rates(self, v_l, v_r):
        """
        Set the drive rates (angular velocities) for this robot's wheels in rad/s
        :param v_l: Velocity of left wheel
        :param v_r: Velocity of right while
        """
        # simulate physical limit on drive motors
        v_l = min(self.robot_cfg["wheel"]["max_speed"], v_l)
        v_r = min(self.robot_cfg["wheel"]["max_speed"], v_r)

        # set drive rates
        self.left_wheel_drive_rate = v_l
        self.right_wheel_drive_rate = v_r
