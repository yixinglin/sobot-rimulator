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


from supervisor.ControlState import *
from simulation.exceptions import GoalReachedException, GoalNotReachedException
from utils import linalg2_util as linalg
from supervisor.Timer import Timer

class SupervisorStateMachine:

    def __init__(self, supervisor, control_config):
        """
        Initializes a SupervisorStateMachine object
        :param supervisor: The underlying supervisor
        :param control_config: The control configuration
        """
        self.supervisor = supervisor

        # initialize state
        self.transition_to_state_go_to_goal()

        # progress tracking
        self.best_distance_to_goal = float("inf")

        self.cfg = control_config

        self.timer = Timer()

    def update_state(self):
        """
        Update the control state
        """
        if self.current_state == ControlState.GO_TO_GOAL:
            self.execute_state_go_to_goal()
        elif self.current_state == ControlState.AVOID_OBSTACLES:
            self.execute_state_avoid_obstacles()
        elif self.current_state == ControlState.SLIDE_LEFT:
            self.execute_state_slide_left()
        elif self.current_state == ControlState.SLIDE_RIGHT:
            self.execute_state_slide_right()
        elif self.current_state == ControlState.AT_GOAL:
            self.execute_state_go_to_goal()
        else:
            raise Exception("undefined supervisor state or behavior", self.current_state)

        # change the goal if robot cannot reach it in time
        if self.timer.not_reach_goal_in_time(self.supervisor.goal):
            self.transition_to_state_go_to_goal()
            self.timer.reset()
            print ("GoalNotReachedException: Not reach the goal in time.")
            raise GoalNotReachedException()  # add a new goal not far from the robot

    # === STATE PROCEDURES ===
    def execute_state_go_to_goal(self):
        if self.condition_at_goal():
            self.transition_to_state_at_goal()
        elif self.condition_danger():
            self.transition_to_state_avoid_obstacles()
        elif self.condition_at_obstacle():
            sl = self.condition_slide_left()
            sr = self.condition_slide_right()
            if sl and not sr:
                self.transition_to_state_slide_left()
            elif sr and not sl:
                self.transition_to_state_slide_right()
            # elif sl and sr: raise Exception( "cannot determine slide direction" )

    def execute_state_avoid_obstacles(self):
        if self.condition_at_goal():
            self.transition_to_state_at_goal()
        elif not self.condition_danger():
            sl = self.condition_slide_left()
            sr = self.condition_slide_right()
            if sl and not sr:
                self.transition_to_state_slide_left()
            elif sr and not sl:
                self.transition_to_state_slide_right()
            elif not sr and not sl:
                self.transition_to_state_go_to_goal()
            # else: raise Exception( "cannot determine slide direction" )

    def execute_state_slide_left(self):
        if self.condition_at_goal():
            self.transition_to_state_at_goal()
        elif self.condition_danger():
            self.transition_to_state_avoid_obstacles()
        elif self.condition_progress_made() and \
            (self.timer.previous_best_distance_not_changed_in_time(self.best_distance_to_goal)
                or not self.condition_slide_left()):
            self.transition_to_state_go_to_goal()

    def execute_state_slide_right(self):
        if self.condition_at_goal():
            self.transition_to_state_at_goal()
        elif self.condition_danger():
            self.transition_to_state_avoid_obstacles()
        elif self.condition_progress_made() and \
            (self.timer.previous_best_distance_not_changed_in_time(self.best_distance_to_goal)
             or not self.condition_slide_right()):
            self.transition_to_state_go_to_goal()

    # def execute_state_gtg_and_ao( self ):
    #   if self.condition_at_goal():        self.transition_to_state_at_goal()
    #   elif self.condition_danger():       self.transition_to_state_avoid_obstacles()
    #   elif self.condition_no_obstacle():  self.transition_to_state_go_to_goal()

    # === STATE TRANSITIONS ===
    def transition_to_state_at_goal(self):
        self.current_state = ControlState.AT_GOAL
        raise GoalReachedException()

    def transition_to_state_avoid_obstacles(self):
        self.current_state = ControlState.AVOID_OBSTACLES
        self.supervisor.current_controller = self.supervisor.avoid_obstacles_controller

    def transition_to_state_go_to_goal(self):
        self.current_state = ControlState.GO_TO_GOAL
        self.supervisor.current_controller = self.supervisor.go_to_goal_controller

    def transition_to_state_slide_left(self):
        self.current_state = ControlState.SLIDE_LEFT
        self._update_best_distance_to_goal()
        self.supervisor.current_controller = self.supervisor.follow_wall_controller

    def transition_to_state_slide_right(self):
        self.current_state = ControlState.SLIDE_RIGHT
        self._update_best_distance_to_goal()
        self.supervisor.current_controller = self.supervisor.follow_wall_controller

    def transition_to_state_gtg_and_ao(self):
        self.current_state = ControlState.GTG_AND_AO
        self.supervisor.current_controller = self.supervisor.gtg_and_ao_controller

    # === CONDITIONS ===
    def condition_at_goal(self):
        return linalg.distance(self.supervisor.estimated_pose.vposition(), self.supervisor.goal) < self.cfg["goal_reached_distance"]

    def condition_at_obstacle(self):
        for d in self._forward_sensor_distances():
            if d < self.cfg["caution_distance"]:
                return True
        return False

    def condition_danger(self):
        for d in self._forward_sensor_distances():
            if d < self.cfg["danger_distance"]:
                return True
        return False

    def condition_no_obstacle(self):
        for d in self._forward_sensor_distances():
            if d < self.cfg["caution_distance"]:
                return False
        return True

    def condition_progress_made(self):
        return self._distance_to_goal() < self.best_distance_to_goal - self.cfg["progress_epsilon"]

    def condition_slide_left(self):
        heading_gtg = self.supervisor.go_to_goal_controller.gtg_heading_vector
        heading_ao = self.supervisor.avoid_obstacles_controller.ao_heading_vector
        heading_fwl = self.supervisor.follow_wall_controller.l_fw_heading_vector

        ao_cross_fwl = linalg.cross(heading_ao, heading_fwl)
        fwl_cross_gtg = linalg.cross(heading_fwl, heading_gtg)
        ao_cross_gtg = linalg.cross(heading_ao, heading_gtg)

        return ((ao_cross_gtg > 0.0 and ao_cross_fwl > 0.0 and fwl_cross_gtg > 0.0) or
                (ao_cross_gtg <= 0.0 and ao_cross_fwl <= 0.0 and fwl_cross_gtg <= 0.0))

    def condition_slide_right(self):
        heading_gtg = self.supervisor.go_to_goal_controller.gtg_heading_vector
        heading_ao = self.supervisor.avoid_obstacles_controller.ao_heading_vector
        heading_fwr = self.supervisor.follow_wall_controller.r_fw_heading_vector

        ao_cross_fwr = linalg.cross(heading_ao, heading_fwr)
        fwr_cross_gtg = linalg.cross(heading_fwr, heading_gtg)
        ao_cross_gtg = linalg.cross(heading_ao, heading_gtg)

        return ((ao_cross_gtg > 0.0 and ao_cross_fwr > 0.0 and fwr_cross_gtg > 0.0) or
                (ao_cross_gtg <= 0.0 and ao_cross_fwr <= 0.0 and fwr_cross_gtg <= 0.0))

    # === helper methods ===
    def _forward_sensor_distances(self):
        return self.supervisor.proximity_sensor_distances[1:7]

    def _distance_to_goal(self):
        return linalg.distance(self.supervisor.estimated_pose.vposition(), self.supervisor.goal)

    def _update_best_distance_to_goal(self):
        self.best_distance_to_goal = min(self.best_distance_to_goal, self._distance_to_goal())