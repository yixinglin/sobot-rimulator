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
from models.Pose import Pose
from models.Polygon import Polygon
from models.obstacles.RectangleObstacle import RectangleObstacle
from models.obstacles.FeaturePoint import FeaturePoint
from utils import linalg2_util as linalg
#seed(42)


class MapManager:

    def __init__(self, map_config):
        """
        Initializes a MapManager object
        :param map_config: The map configuration
        """
        self.current_obstacles = []
        self.current_goal = None
        self.cfg = map_config

    def random_map(self, world):
        """
        Generates a random map and goal
        :param world: The world the map is generated for
        """
        obstacles = []
        if self.cfg["obstacle"]["rectangle"]["enabled"]:
            obstacles += self.__generate_rectangle_obstacles(world)
        if self.cfg["obstacle"]["feature"]["enabled"] \
            and self.cfg["obstacle"]["rectangle"]["enabled"]:
            obstacles += self.__generate_features(world, obstacles)
        if self.cfg["obstacle"]["feature"]["enabled"] \
            and not self.cfg["obstacle"]["rectangle"]["enabled"]:
            obstacles += self.__generate_random_features(world)

        # update the current obstacles and goal
        self.current_obstacles = obstacles
        self.add_new_goal()

        # apply the new obstacles and goal to the world
        self.apply_to_world(world)

    def add_new_goal(self, world = None):
        """
        Adds a new goal
        """
        i = 3000
        max_dist = self.cfg["goal"]["max_distance"]
        while i>0:
            i -= 1
            goal = self.__generate_new_goal()
            intersects = self.__check_obstacle_intersections(goal)
            if not intersects and world is None:
                self.current_goal = goal
                break
            elif not intersects and world is not None: # add a new goal not far from the robot
                rob_x, rob_y = world.robots[0].supervisor.estimated_pose.vposition()
                distance_to_robot = linalg.distance([rob_x, rob_y], goal)
                if distance_to_robot < 1 and distance_to_robot > 0.5: # being able to a new goal not far from the robot
                    self.current_goal = goal
                    break
                if rob_x**2 + rob_y**2 > max_dist**2: # not being able to a new goal near the robot
                    self.current_goal = goal
                    break

    def __generate_random_features(self, world):
        """
        Generate random octagon obstacles
        :param world: The world for which they are generated
        :return: List of generated octagon obstacles
        """
        obs_radius = self.cfg["obstacle"]["feature"]["radius"]
        obs_min_count = self.cfg["obstacle"]["feature"]["min_count"]
        obs_max_count = self.cfg["obstacle"]["feature"]["max_count"]
        obs_min_dist = self.cfg["obstacle"]["feature"]["min_distance"]
        obs_max_dist = self.cfg["obstacle"]["feature"]["max_distance"]

        # generate the obstacles
        obstacles = []
        obs_dist_range = obs_max_dist - obs_min_dist
        num_obstacles = randrange(obs_min_count, obs_max_count + 1)

        test_geometries = [r.global_geometry for r in world.robots]
        while len(obstacles) < num_obstacles:

            # generate position
            dist = obs_min_dist + (random() * obs_dist_range)
            phi = -pi + (random() * 2 * pi)
            x = dist * sin(phi)
            y = dist * cos(phi)

            # generate orientation
            theta = -pi + (random() * 2 * pi)

            # test if the obstacle overlaps the robots or the goal
            obstacle = FeaturePoint(obs_radius, Pose(x, y, theta), 0)
            intersects = False
            for test_geometry in test_geometries:
                intersects |= geometrics.convex_polygon_intersect_test(test_geometry, obstacle.global_geometry)
            if not intersects:
                obstacles.append(obstacle)

        for i, feature in enumerate(obstacles):
            feature.id = i
        return obstacles

    def __generate_rectangle_obstacles(self, world):
        """
        Generate random rectangle obstacles
        :param world: The world for which they are generated
        :return: List of generated rectangle obstacles
        """
        obs_min_dim = self.cfg["obstacle"]["rectangle"]["min_dim"]
        obs_max_dim = self.cfg["obstacle"]["rectangle"]["max_dim"]
        obs_max_combined_dim = self.cfg["obstacle"]["rectangle"]["max_combined_dim"]
        obs_min_count = self.cfg["obstacle"]["rectangle"]["min_count"]
        obs_max_count = self.cfg["obstacle"]["rectangle"]["max_count"]
        obs_min_dist = self.cfg["obstacle"]["rectangle"]["min_distance"]
        obs_max_dist = self.cfg["obstacle"]["rectangle"]["max_distance"]

        # generate the obstacles
        obstacles = []
        obs_dim_range = obs_max_dim - obs_min_dim
        obs_dist_range = obs_max_dist - obs_min_dist
        num_obstacles = randrange(obs_min_count, obs_max_count + 1)

        test_geometries = [r.global_geometry for r in world.robots]
        while len(obstacles) < num_obstacles:
            # generate dimensions
            width = obs_min_dim + (random() * obs_dim_range )
            height = obs_min_dim + (random() * obs_dim_range )
            while width + height > obs_max_combined_dim:
                height = obs_min_dim + (random() * obs_dim_range )

            # generate position
            dist = obs_min_dist + (random() * obs_dist_range)
            phi = -pi + (random() * 2 * pi)
            x = dist * sin(phi)
            y = dist * cos(phi)

            # generate orientation
            theta = -pi + (random() * 2 * pi)

            # test if the obstacle overlaps the robots or the goal
            obstacle = RectangleObstacle(width, height, Pose(x, y, theta))
            intersects = False
            for test_geometry in test_geometries:
                intersects |= geometrics.convex_polygon_intersect_test(test_geometry, obstacle.global_geometry)
            if not intersects:
                obstacles.append(obstacle)
        return obstacles

    def __generate_feature_line2(self, world, x0, y0, x1, y1):
        r = 8  # resolution: pixs/meter
        obs_radius = 0.04
        line = geometrics.bresenham_line(x0*r, y0*r, x1*r, y1*r)
        test_geometries = [r.global_geometry for r in world.robots]

        obstacles = []
        for x, y in line:
            x = x/r
            y = y/r
            theta = -pi + (random() * 2 * pi)
            obstacle = FeaturePoint(obs_radius, Pose(x, y, theta), 0)

            # intersects = self.__check_obstacle_intersections([x,y])
            # if not intersects:
            #     obstacles.append(obstacle)
            intersects = False
            for test_geometry in test_geometries:
                intersects |= geometrics.convex_polygon_intersect_test(test_geometry, obstacle.global_geometry)
            if not intersects:
                obstacles.append(obstacle)
        return obstacles

    def feature_test_geometry(self, x,y):
        n = 6
        r = 0.3
        goal_test_geometry = []
        for i in range(n):
            goal_test_geometry.append(
                [x + r * cos(i * 2 * pi / n), y + r * sin(i * 2 * pi / n)]
            )
        test_geometry = Polygon(goal_test_geometry)
        return test_geometry


    def __generate_feature_line(self, x0, y0, x1, y1, obs_radius, density):
        c = density  # feature density
        a = atan2((y1-y0), (x1-x0))
        obstacles = []
        x = x0; y = y0

        dx = copysign(c * cos(a), x1 - x0)
        dy = copysign(c * sin(a), y1 - y0)
        length = int((x1 - x0) // dx)
        for i in range(length):
            x += dx
            y += dy
            theta = -pi + (random() * 2 * pi)
            nx = dx*(random()-1)*2
            ny = dy*(random()-1)*2
            feature = FeaturePoint(obs_radius, Pose(x + nx, y + ny, theta), 0)

            obstacles.append(feature)
        return obstacles

    def __generate_feature_obstacle(self, world, vertexes):
        radius = self.cfg["obstacle"]["feature"]["radius"]
        density = self.cfg["obstacle"]["feature"]["density"]
        num_vertexes = len(vertexes)
        obstacles = []
        for i in range(num_vertexes):
            j = 0 if i+1 >= num_vertexes else i+1
            x0 = vertexes[i][0]
            y0 = vertexes[i][1]
            x1 = vertexes[j][0]
            y1 = vertexes[j][1]
            obstacles += self.__generate_feature_line(x0, y0, x1, y1, radius, density)
            #obstacles.pop()
        return obstacles

    def __generate_features(self, world, obstacles):
        features = []
        for obstacle in obstacles:
            if type(obstacle) == FeaturePoint:
                continue
            features += self.__generate_feature_obstacle(world, obstacle.global_geometry.vertexes)
        # add identifiers for each feature.
        for i, f in enumerate(features):
            f.id = i

        print ("#Feature: ", len(features))
        return features

    def __generate_new_goal(self):
        """
        Generate a new random goal
        :return: The generated goal
        """
        min_dist = self.cfg["goal"]["min_distance"]
        max_dist = self.cfg["goal"]["max_distance"]
        goal_dist_range = max_dist - min_dist
        dist = min_dist + (random() * goal_dist_range)
        phi = -pi + (random() * 2 * pi)
        x = dist * sin(phi)
        y = dist * cos(phi)
        goal = [x, y]
        return goal

    def __check_obstacle_intersections(self, goal):
        """
        Check for intersections between the goal and the obstacles
        :param goal: The goal posibition
        :return: Boolean value indicating if the goal is too close to an obstacle
        """
        # generate a proximity test geometry for the goal
        min_clearance = self.cfg["goal"]["min_clearance"]
        n = 6   # goal is n sided polygon
        goal_test_geometry = []
        for i in range(n):
            goal_test_geometry.append(
                [goal[0] + min_clearance * cos(i * 2 * pi / n),
                 goal[1] + min_clearance * sin(i * 2 * pi / n)])
        goal_test_geometry = Polygon(goal_test_geometry)
        intersects = False
        for obstacle in self.current_obstacles:
            intersects |= geometrics.convex_polygon_intersect_test(goal_test_geometry, obstacle.global_geometry)
        return intersects

    def save_map(self, filename):
        """
        Save the map, including obstacles and goal, as well as the current random state to enable reproducibility
        :param filename: The filename under which the map shall be stored
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.current_obstacles, file)
            pickle.dump(self.current_goal, file)
            pickle.dump(getstate(), file)

    def load_map(self, filename):
        """
        Load a map from the file
        :param filename: Filename from which the map shall be loaded
        """
        with open(filename, 'rb') as file:
            self.current_obstacles = pickle.load(file)
            self.current_goal = pickle.load(file)
            try:
                setstate(pickle.load(file))
            except EOFError:
                print("No random state stored")

    def apply_to_world(self, world):
        """
        Apply the current obstacles and goal to the world
        :param world: The world that shall be updated
        """
        # add the current obstacles
        for obstacle in self.current_obstacles:
            world.add_obstacle(obstacle)

        # program the robot supervisors
        for robot in world.robots:
            robot.supervisor.goal = self.current_goal[:]
