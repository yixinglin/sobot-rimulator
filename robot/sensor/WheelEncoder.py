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


from robot.sensor.Sensor import *


class WheelEncoder(Sensor):

    def __init__(self, ticks_per_rev):
        """
        Initializes a WheelEncoder object that provides odometric information
        :param ticks_per_rev: Specifies the accuracy per wheel revolution as amount of ticks.
        """
        self.ticks_per_rev = ticks_per_rev
        self.real_revs = 0.0
        self.tick_count = 0

    def step_revolutions(self, revolutions):
        """
        # Update the tick count for this wheel encoder
        :param revolutions: A float specifiying the number of forward revolutions made
        """
        self.real_revs += revolutions
        self.tick_count = int(self.real_revs * self.ticks_per_rev)

    def read(self):
        """
        :return: The current wheel encoder reading
        """
        return self.tick_count
