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


from plotters.ProximitySensorPlotter import *
from plotters.SupervisorPlotter import *


class RobotPlotter:

    def __init__(self, robot):
        """
        Initializes a RobotPlotter object
        :param robot: The underlying robot
        """
        self.robot = robot
        self.robot_shape = robot.robot_cfg["top_plate"]

        # add the supervisor plotter for this robot
        self.supervisor_plotter = SupervisorPlotter(robot.supervisor, robot.geometry)

        # add the IR sensor views for this robot
        self.ir_sensor_plotters = []
        for ir_sensor in robot.ir_sensors:
            self.ir_sensor_plotters.append(ProximitySensorPlotter(ir_sensor, radians(robot.robot_cfg["sensor"]["cone_angle"])))

        self.traverse_path = []  # this robot's traverse path

    def draw_robot_to_frame(self, frame, draw_invisibles=False):
        """
        Draws a robot to the frame
        :param frame: The frame to be used
        :param draw_invisibles: Boolean value specifying whether invisibles shall be drawn
        """
        # update the robot traverse path
        position = self.robot.pose.vposition()
        self.traverse_path.append(position)

        # draw the internal state ( supervisor ) to the frame
        self.supervisor_plotter.draw_supervisor_to_frame(frame, draw_invisibles)

        # draw the IR sensors to the frame if indicated
        if draw_invisibles:
            for ir_sensor_plotter in self.ir_sensor_plotters:
                ir_sensor_plotter.draw_proximity_sensor_to_frame(frame)

        # draw the robot
        robot_bottom = self.robot.global_geometry.vertexes
        frame.add_polygons([robot_bottom],
                           color="blue",
                           alpha=0.5)
        # add decoration
        robot_pos, robot_theta = self.robot.pose.vunpack()
        robot_top = linalg.rotate_and_translate_vectors(self.robot_shape, robot_theta, robot_pos)
        frame.add_polygons([robot_top],
                           color="black",
                           alpha=0.5)

        # draw the robot's traverse path if indicated
        if draw_invisibles:
            self._draw_real_traverse_path_to_frame(frame)

    def _draw_real_traverse_path_to_frame(self, frame):
        """
        Draws the real traverse path of the robot to the frame
        :param frame: The frame to be used
        """
        frame.add_lines([self.traverse_path],
                        color="black",
                        linewidth=0.01)
