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
from models.obstacles.CircleObstacle import CircleObstacle
from robot.Robot import Robot
from models.Pose import Pose
from plotters.ObstaclePlotter import *
from plotters.RobotPlotter import *
import numpy as np


class SlamPlotter:

    def __init__(self, slam, viewer, radius, robot_config):
        self.slam = slam
        self.viewer = viewer
        self.robot_bottom_shape = robot_config["bottom_plate"]
        self.robot_top_shape = robot_config["top_plate"]
        self.radius = radius

    def draw_slam_to_frame(self):
        frame = self.viewer.current_frames[1]
        self.__draw_robot_to_frame(frame, self.slam.get_estimated_pose())

        # draw all the obstacles
        for landmark in self.slam.get_landmarks():
            obstacle = CircleObstacle(self.radius, Pose(landmark[0], landmark[1], 0))
            obstacle_view = ObstaclePlotter(obstacle)
            obstacle_view.draw_obstacle_to_frame(self.viewer.current_frames[1])

        if self.viewer.draw_invisibles:
            # Plot variance ellipses
            self.__draw_confidence_ellipse(frame)
            #vars = self.slam.get_covariances().diagonal()
            #std_dev_pos = np.sqrt(vars[0:2])
            #frame.add_ellipse(self.slam.get_estimated_pose().sunpack(), 0, std_dev_pos[0], std_dev_pos[1], color="red", alpha=0.5)

    def __draw_robot_to_frame(self, frame, robot_pose, draw_invisibles=False):
        robot_pos, robot_theta = robot_pose.vunpack()
        # draw the robot
        robot_bottom = linalg.rotate_and_translate_vectors(self.robot_bottom_shape, robot_theta, robot_pos)
        frame.add_polygons([robot_bottom],
                           color="blue",
                           alpha=0.5)
        # add decoration
        robot_top = linalg.rotate_and_translate_vectors(self.robot_top_shape, robot_theta, robot_pos)
        frame.add_polygons([robot_top],
                           color="black",
                           alpha=0.5)

    def __draw_confidence_ellipse(self, frame, n_std=1.0):
        cov = self.slam.get_covariances()[:2, :2]  # Get covariances of position arguments
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        eigvals, eigvecs = np.linalg.eig(cov)
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        print(eigvals, ell_radius_x, ell_radius_y)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        scale_y = np.sqrt(cov[1, 1]) * n_std
        ell_radius_x *= scale_x
        ell_radius_y *= scale_y
        angle = pi/4

        frame.add_ellipse(self.slam.get_estimated_pose().sunpack(),
                          angle, ell_radius_x, ell_radius_y,
                          color="red", alpha=0.5)

        """
        ellipse = Ellipse((0, 0),
                          width=ell_radius_x * 2,
                          height=ell_radius_y * 2,
                          facecolor=facecolor,
                          **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)
        """
