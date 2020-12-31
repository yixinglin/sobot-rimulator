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

from utils import linalg2_util as linalg
from supervisor.controllers.Controller import Controller

class GoToGoalController(Controller):

    def __init__(self, supervisor):
        """
        Initializes a GoToGoalController object
        :param supervisor: The underlying supervisor
        """
        # bind the supervisor
        self.supervisor = supervisor

        # gains
        self.kP = 5.0
        self.kI = 0.0
        self.kD = 0.0

        # stored values - for computing next results
        self.prev_time = 0.0
        self.prev_eP = 0.0
        self.prev_eI = 0.0

        # key vectors and data (initialize to any non-zero vector)
        self.gtg_heading_vector = [1.0, 0.0]

    def update_heading(self):
        """
        Generate and store new heading vector
        """
        self.gtg_heading_vector = self.calculate_gtg_heading_vector()

    def execute(self):
        """
        Executes the controllers update during one simulation cycle
        """
        # calculate the time that has passed since the last control iteration
        current_time = self.supervisor.time()
        dt = current_time - self.prev_time

        # calculate the error terms
        theta_d = atan2(self.gtg_heading_vector[1], self.gtg_heading_vector[0])
        eP = theta_d
        eI = self.prev_eI + eP * dt
        eD = (eP - self.prev_eP) / dt

        # calculate angular velocity
        omega = self.kP * eP + self.kI * eI + self.kD * eD

        # calculate translational velocity
        # velocity is v_max when omega is 0,
        # drops rapidly to zero as |omega| rises
        v = self.supervisor.v_max() / (abs(omega) + 1) ** 0.5

        # store values for next control iteration
        self.prev_time = current_time
        self.prev_eP = eP
        self.prev_eI = eI

        self.supervisor.set_outputs(v, omega)

    def calculate_gtg_heading_vector(self):
        """
        :return: A go-to-goal heading vector in the robot's reference frame
        """
        # get the inverse of the robot's pose
        robot_inv_pos, robot_inv_theta = self.supervisor.estimated_pose().inverse().vunpack()

        # calculate the goal vector in the robot's reference frame
        goal = self.supervisor.goal()
        goal = linalg.rotate_and_translate_vector(goal, robot_inv_theta, robot_inv_pos)

        return goal
