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


from math import *

from models.LineSegment import *
from models.Pose import *
from robot.sensor.Sensor import *


class ProximitySensor(Sensor):

    def __init__(self, robot,  # robot this sensor is attached to
                 placement_pose,
                 # pose of this sensor relative to the robot (NOTE: normalized on robot located at origin and with theta 0, i.e. facing east )
                 sensor_config, id):
        """
        Initializes a ProximitySensor object
        :param robot: THe underlying robot
        :param placement_pose: The pose at which the sensor is placed relative to the robot
        :param sensor_config: The sensor configurations
        :param id: The sensor ID
        """
        # bind the robot
        self.robot = robot
        self.id = id # sensor ID
        # pose attributes
        self.placement_pose = placement_pose  # pose of this sensor relative to the robot
        self.pose = Pose(0.0, 0.0, 0.0)  # global pose of this sensor

        # sensitivity attributes
        self.min_range = sensor_config["min_range"]
        self.max_range = sensor_config["max_range"]
        self.min_read_value = sensor_config["min_read_value"]
        self.max_read_value = sensor_config["max_read_value"]

        # detector line
        self.detector_line_source = LineSegment([[0.0, 0.0], [self.max_range, 0.0]])
        self.detector_line = LineSegment([[0.0, 0.0], [self.max_range, 0.0]])

        # pose and detector_line are incorrect until:
        # set initial position
        self.update_position()

        # physical distance detected to target as a proportion of max_range ( must be in range [0, 1] or None )
        self.target_delta = None

        # sensor output
        self.read_value = self.min_read_value

        # factor used for converting measured distance into a read value
        self.factor = log(self.min_read_value / self.max_read_value) / (self.max_range - self.min_range)

    def detect(self, delta):
        """
        Set this proximity sensor to detect an object at distance ( delta * max_range )
        :param delta: Factor of the maximum sensor range that was detected
        """
        if delta is not None and (delta < 0.0 or delta > 1.0):
            raise Exception("delta out of bounds - must be in range [0.0, 1.0]")

        if delta is None:
            self.target_delta = None
            self.read_value = self.min_read_value
        else:
            max_range = self.max_range
            min_range = self.min_range

            d = max_range * delta  # d is the real distance in meters
            if d <= min_range:  # d in [0.00, 0.02]
                self.target_delta = min_range / max_range
                self.read_value = self.max_read_value
            else:  # d in (0.02, 0.20]
                self.target_delta = delta
                self.read_value = max(self.min_read_value,
                                      int(ceil(self.max_read_value * e ** (self.factor * (d - min_range))))
                                      )

    def read(self):
        """
        :return: The current sensor reading
        """
        return self.read_value

    def update_position(self):
        """
        Update the global position of this sensor
        """
        # update global pose
        self._update_pose()

        # update detector line
        self.detector_line = self.detector_line_source.get_transformation_to_pose(self.pose)

    def _update_pose(self):
        """
        Update this sensor's pose
        """
        self.pose = self.placement_pose.transform_to(self.robot.pose)
