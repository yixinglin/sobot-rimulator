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
from models.obstacles.OctagonObstacle import OctagonObstacle
from models.Pose import Pose
from plotters.ObstaclePlotter import *
from plotters.RobotPlotter import *
from matplotlib import pyplot as plt
import numpy as np

from supervisor.slam.EKFSlam import EKFSlam
from supervisor.slam.FastSlam import FastSlam


class SlamPlotter:

    def __init__(self, slam, viewer, radius, robot_config, frame_number):
        """
        Initializes a SlamPlotter object
        :param slam: The underlying slam algorithm object
        :param viewer: The viewer to be used
        :param radius: The radius with which the estimated obstacles shall be drawn
        :param robot_config: The robot configuration
        :param frame_number: The frame number to be used
        """
        self.slam = slam
        self.viewer = viewer
        self.frame_number = frame_number
        self.robot_bottom_shape = robot_config["bottom_plate"]
        self.robot_top_shape = robot_config["top_plate"]
        self.radius = radius

    def draw_slam_to_frame(self):
        """
        Draws a SLAM visualization to the frame
        """
        frame = self.viewer.current_frames[self.frame_number]
        self.__draw_robot_to_frame(frame, self.slam.get_estimated_pose())

        # draw all the obstacles
        for landmark in self.slam.get_landmarks():
            frame.add_circle((landmark[0], landmark[1]), self.radius, "darkgreen", alpha=0.8)

        if self.viewer.draw_invisibles and isinstance(self.slam, EKFSlam):
            self.__draw_confidence_ellipse(frame)

        if self.viewer.draw_invisibles and isinstance(self.slam, FastSlam):
            self.__draw_particles(frame, self.radius/2)


    def __draw_particles(self, frame, radius, color = "blue"):
        for x, y, w in self.slam.get_particles():
            frame.add_circle((x, y), radius, color, alpha=0.1)


    def plot_covariances(self):
        """
        Plots the covariance matrix
        """
        cov = self.slam.get_covariances()
        plt.matshow(cov)
        plt.show()

    def __draw_robot_to_frame(self, frame, robot_pose):
        """
        Draws the roboto to the frame
        :param frame: The frame to be used
        :param robot_pose: The pose of the robot
        """
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

    def __draw_confidence_ellipse(self, frame):
        """
        Draws confidence ellipses based on the covariance matrix to the frame. Only supported for EKFSLAM.
        :param frame: The frame to be used
        """
        cov = self.slam.get_covariances()[:2, :2]  # Get covariances of position arguments
        eigvals, eigvecs = np.linalg.eig(cov)
        if eigvals[0] < eigvals[1]:
            eigvals = eigvals[::-1]  # Swap eigenvalues as well as eigenvectors
            eigvecs = eigvecs[:, ::-1]
        angle = atan2(eigvecs[1, 0], eigvecs[0, 0])

        frame.add_ellipse(self.slam.get_estimated_pose().sunpack(),
                          angle, eigvals[0], eigvals[1],
                          color="red", alpha=0.5)


class GraphSlamPlotter(SlamPlotter):

    def __init__(self, slam, viewer, radius, robot_config, frame_number):
        SlamPlotter.__init__(self, slam, viewer, radius, robot_config, frame_number)
        self.count = 0

    def draw_slam_to_frame(self):
        SlamPlotter.draw_slam_to_frame(self)
        frame = self.viewer.current_frames[self.frame_number]
        self.count += 1
        if self.slam.draw_trajectory:
            self.robot_estimated_traverse_path = self.slam.get_estimated_trajectory()
            self.__draw_robot_estimated_traverse_path_to_frame(frame)

    def __draw_robot_estimated_traverse_path_to_frame(self, frame):
        # Draw traverse path of the robot estimated by the Graph-based SLAM
        frame.add_lines([self.robot_estimated_traverse_path],
                        linewidth=0.01,
                        color="black",
                        alpha=0.8)

    def plot_graph(self):
        fig = plt.figure()
        fig.add_subplot(131)
        self.slam.graph.draw()
        plt.title("Graph")
        plt.legend()
        plt.axis('square')

        fig.add_subplot(132)
        H, _ = self.slam.get_hessian()
        H = np.abs(H)
        plt.matshow(H, cmap="Greys", fignum=False)
        plt.title("Hessian Matrix (Intensity)")
        H = H / np.max(H)
        H[H > 0] = 1
        fig.add_subplot(133)
        plt.matshow(H, cmap="Greys", fignum=False)
        plt.title("Hessian Matrix (Binary)")

        plt.show()
