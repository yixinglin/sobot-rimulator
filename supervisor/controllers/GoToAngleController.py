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


from utils import math_util
from supervisor.controllers.Controller import Controller

class GoToAngleController(Controller):

    def __init__(self, supervisor):
        """
        Initializes a GoToAngleController
        :param supervisor: The underlying supervisor
        """
        # bind the supervisor
        self.supervisor = supervisor

        # gains
        self.k_p = 5.0

    def execute(self, theta_d):
        """
        Executes the controllers update during one simulation cycle
        :param theta_d: The angle in which the robot should drive
        """
        theta = self.supervisor.estimated_pose().theta
        e = math_util.normalize_angle(theta_d - theta)
        omega = self.k_p * e

        self.supervisor.set_outputs(1.0, omega)
