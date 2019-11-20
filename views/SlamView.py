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
from models.rectangle_obstacle import RectangleObstacle
from models.robot import Robot
from models.pose import Pose
from views.obstacle_view import *
from views.robot_view import *

MAJOR_GRIDLINE_INTERVAL = 1.0  # meters
MAJOR_GRIDLINE_SUBDIVISIONS = 5  # minor gridlines for every major gridline


class SlamView:

    def __init__(self, slam, viewer):
        # bind the viewer
        self.slam = slam
        self.viewer = viewer
        self.robot = Robot()

    def draw_slam_to_frame(self):
        self.__draw_robot_to_frame(self.viewer.current_frames[1], self.slam.get_estimated_pose())

        # draw all the obstacles
        for landmark in self.slam.get_landmarks():
            obstacle = RectangleObstacle(0.05, 0.05, Pose(landmark[0], landmark[1], 0))
            obstacle_view = ObstacleView(obstacle)
            obstacle_view.draw_obstacle_to_frame(self.viewer.current_frames[1])

    def __draw_robot_to_frame(self, frame, robot_pose, draw_invisibles=False):
        # draw the robot
        robot_bottom = self.robot.geometry.get_transformation_to_pose(robot_pose).vertexes
        frame.add_polygons([robot_bottom],
                           color="blue",
                           alpha=0.5)
        # add decoration
        robot_pos, robot_theta = robot_pose.vunpack()
        robot_top = linalg.rotate_and_translate_vectors(K3_TOP_PLATE, robot_theta, robot_pos)
        frame.add_polygons([robot_top],
                           color="black",
                           alpha=0.5)

    def add_landmark(self, landmark):
        obstacle = RectangleObstacle(0.05, 0.05, Pose(landmark[0], landmark[1], 0))
        obstacle_view = ObstacleView(obstacle)
        self.obstacle_views.append(obstacle_view)
