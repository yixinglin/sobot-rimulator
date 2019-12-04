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

import pickle
from math import *
from random import *

import utils.geometrics_util as geometrics
from models.pose import Pose
from models.rectangle_obstacle import *

# random environment parameters
OBS_MIN_DIM = 0.05  # meters
OBS_MAX_DIM = 0.075  # meters
OBS_MAX_COMBINED_DIM = 0.125  # meters
OBS_MIN_COUNT = 25
OBS_MAX_COUNT = 50
OBS_MIN_DIST = 0.4  # meters
OBS_MAX_DIST = 2  # meters
GOAL_MIN_DIST = 0  # meters
GOAL_MAX_DIST = 2  # meters
MIN_GOAL_CLEARANCE = 0.2  # meters


class MapManager:

    def __init__(self):
        self.current_obstacles = []
        self.current_goal = None

    def random_map(self, world):
        # OBSTACLE PARAMS
        obs_min_dim = OBS_MIN_DIM
        obs_max_dim = OBS_MAX_DIM
        obs_max_combined_dim = OBS_MAX_COMBINED_DIM
        obs_min_count = OBS_MIN_COUNT
        obs_max_count = OBS_MAX_COUNT
        obs_min_dist = OBS_MIN_DIST
        obs_max_dist = OBS_MAX_DIST

        # BUILD RANDOM ELEMENTS

        # generate the obstacles
        obstacles = []
        obs_dim_range = obs_max_dim - obs_min_dim
        obs_dist_range = obs_max_dist - obs_min_dist
        num_obstacles = randrange(obs_min_count, obs_max_count + 1)

        test_geometries = [r.global_geometry for r in world.robots]
        while len(obstacles) < num_obstacles:

            # generate dimensions
            width = obs_min_dim + (random() * obs_dim_range)
            height = obs_min_dim + (random() * obs_dim_range)
            while width + height > obs_max_combined_dim:
                height = obs_min_dim + (random() * obs_dim_range)

            # generate position
            dist = obs_min_dist + (random() * obs_dist_range)
            phi = -pi + (random() * 2 * pi)
            x = dist * sin(phi)
            y = dist * cos(phi)

            # generate orientation
            theta = -pi + (random() * 2 * pi)

            # test if the obstacle overlaps the robots or the goal
            obstacle = RectangleObstacle(width, height,
                                         Pose(x, y, theta))
            intersects = False
            for test_geometry in test_geometries:
                intersects |= geometrics.convex_polygon_intersect_test(test_geometry, obstacle.global_geometry)
            if not intersects:
                obstacles.append(obstacle)

        # update the current obstacles and goal
        self.current_obstacles = obstacles
        self.add_new_goal()

        # apply the new obstacles and goal to the world
        self.apply_to_world(world)

    def add_new_goal(self):
        while True:
            goal = self.__generate_new_goal()
            intersects = self.__check_obstacle_intersections(goal)
            if not intersects:
                self.current_goal = goal
                break

    def __generate_new_goal(self):
        goal_dist_range = GOAL_MAX_DIST - GOAL_MIN_DIST
        dist = GOAL_MIN_DIST + (random() * goal_dist_range)
        phi = -pi + (random() * 2 * pi)
        x = dist * sin(phi)
        y = dist * cos(phi)
        goal = [x, y]
        return goal

    def __check_obstacle_intersections(self, goal):
        # generate a proximity test geometry for the goal
        n = 6   # goal is n sided polygon
        goal_test_geometry = []
        for i in range(n):
            goal_test_geometry.append(
                [goal[0] + MIN_GOAL_CLEARANCE * cos(i * 2 * pi / n),
                 goal[1] + MIN_GOAL_CLEARANCE * sin(i * 2 * pi / n)])
        goal_test_geometry = Polygon(goal_test_geometry)
        intersects = False
        for obstacle in self.current_obstacles:
            intersects |= geometrics.convex_polygon_intersect_test(goal_test_geometry, obstacle.global_geometry)
        return intersects

    def save_map(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.current_obstacles, file)
            pickle.dump(self.current_goal, file)

    def load_map(self, filename):
        with open(filename, 'rb') as file:
            self.current_obstacles = pickle.load(file)
            self.current_goal = pickle.load(file)

    def apply_to_world(self, world):
        # add the current obstacles
        for obstacle in self.current_obstacles:
            world.add_obstacle(obstacle)

        # program the robot supervisors
        for robot in world.robots:
            robot.supervisor.goal = self.current_goal[:]
