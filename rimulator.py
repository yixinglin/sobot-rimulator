#!/usr/bin/env python

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

import sys
import yaml
import gi
from gi.repository import GLib

from plotters.SlamPlotter import *
from supervisor.slam.SlamEvaluation import SlamEvaluation


from plotters.MappingPlotter import MappingPlotter

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk

import gui.Frame
import gui.Viewer

from simulation.MapManager import *
from robot.Robot import *
from robot.RobotSupervisorInterface import *
from simulation.World import *

from plotters.WorldPlotter import *
from simulation.exceptions import CollisionException


class Simulator:

    def __init__(self, cfg):
        """
        Initializes a Simulator object
        :param cfg: The simulators configuration
        """
        # create the GUI
        self.num_frames = 1
        if cfg["slam"]["ekf_slam"]["enabled"]:
            self.num_frames += 1
        if cfg["slam"]["fast_slam"]["enabled"]:
            self.num_frames += 1
        if cfg["slam"]["graph_based_slam"]["enabled"]:
            self.num_frames += 1

        self.viewer = gui.Viewer.Viewer(self, cfg["viewer"], self.num_frames, cfg["slam"])
        self.ekfslam_plotter = None
        self.fastslam_plotter = None
        self.graphbasedslam_plotter = None
        self.ekfslam_evaluation = None
        self.fastslam_evaluation = None
        self.graphbasedslam_evaluation = None

        self.ekfslam_mapping_plotter = None
        self.fastslam_mapping_plotter = None
        self.graphbasedslam_mapping_plotter = None

        self.world_plotter = None

        # create the map manager
        self.map_manager = MapManager(cfg["map"])
        self.world = None

        # timing control
        self.period = cfg["period"]

        # Counts the number of simulation cycles
        self.num_cycles = 0

        self.cfg = cfg

        # gtk simulation event source - for simulation control
        self.sim_event_source = GLib.idle_add(self.initialize_sim, True)  # we use this opportunity to initialize the sim

        # start gtk
        gtk.main()

    def initialize_sim(self, random=False):
        """
        Initializes the simulated world
        :param random: Boolean value specifying if a random map shall be generated
        """
        # reset the viewer
        self.viewer.control_panel_state_init()

        # create the simulation world
        self.world = World(self.period)

        # create the robot
        robot = Robot(self.cfg["robot"])
        # Assign supervisor to the robot
        robot.supervisor = Supervisor(RobotSupervisorInterface(robot, self.cfg['robot']), self.cfg)
        self.world.add_robot(robot)

        # generate a random environment
        if random:
            self.map_manager.random_map(self.world)
        else:
            self.map_manager.apply_to_world(self.world)

        # create the world view
        self.world_plotter = WorldPlotter(self.world, self.viewer)
        n_frame = 1
        if cfg["slam"]["ekf_slam"]["enabled"]:
            self.ekfslam_plotter = SlamPlotter(self.world.supervisors[0].ekfslam, self.viewer, 0.04, self.cfg["robot"], n_frame)
            if cfg["slam"]["mapping"]["enabled"]:
                self.ekfslam_mapping_plotter = MappingPlotter(self.world.supervisors[0].ekfslam_mapping, self.viewer, n_frame)
            if self.cfg["slam"]["evaluation"]["enabled"]:
                self.ekfslam_evaluation = SlamEvaluation(self.world.supervisors[0].ekfslam, self.cfg["slam"]["evaluation"], self.world.robots[0])
            n_frame += 1

        if cfg["slam"]["fast_slam"]["enabled"]:
            self.fastslam_plotter = SlamPlotter(self.world.supervisors[0].fastslam, self.viewer, 0.04, self.cfg["robot"], n_frame)
            if cfg["slam"]["mapping"]["enabled"]:
                self.fastslam_mapping_plotter = MappingPlotter(self.world.supervisors[0].fastslam_mapping, self.viewer, n_frame)
            if self.cfg["slam"]["evaluation"]["enabled"]:
                self.fastslam_evaluation = SlamEvaluation(self.world.supervisors[0].fastslam, self.cfg["slam"]["evaluation"], self.world.robots[0])
            n_frame += 1

        if cfg["slam"]["graph_based_slam"]["enabled"]:
            self.graphbasedslam_plotter = GraphSlamPlotter(self.world.supervisors[0].graphbasedslam, self.viewer, 0.04, self.cfg["robot"], n_frame)
            if cfg["slam"]["mapping"]["enabled"]:
                self.graphbasedslam_mapping_plotter = MappingPlotter(self.world.supervisors[0].graphbasedslam_mapping, self.viewer, n_frame)
            if self.cfg["slam"]["evaluation"]["enabled"]:
                self.graphbasedslam_evaluation = SlamEvaluation(self.world.supervisors[0].graphbasedslam,
                                                                self.cfg["slam"]["evaluation"], self.world.robots[0])
            n_frame += 1

        # register mapping plotters to the system
        self.reg_mapping_plotters = [self.ekfslam_mapping_plotter, self.fastslam_mapping_plotter,
                            self.graphbasedslam_mapping_plotter]
        # register slam plotters to the system
        self.reg_slam_plotters = [self.ekfslam_plotter, self.fastslam_plotter, self.graphbasedslam_plotter]

        # render the initial world
        self.draw_world()

    def play_sim(self):
        """
        Start or continue the simulation
        """
        GLib.source_remove(
            self.sim_event_source)  # this ensures multiple calls to play_sim do not speed up the simulator
        self._run_sim()
        self.viewer.control_panel_state_playing()

    def pause_sim(self):
        """
        Pause the simulation
        """
        GLib.source_remove(self.sim_event_source)
        self.viewer.control_panel_state_paused()

    def step_sim_once(self):
        """
        Progress the simulation by exactly one simulation cycle
        """
        self.pause_sim()
        self._step_sim()

    def end_sim(self, alert_text=''):
        """
        End the simulation
        :param alert_text: Test to be displayed to the user
        """
        GLib.source_remove(self.sim_event_source)
        self.viewer.control_panel_state_finished(alert_text)

    def reset_sim(self):
        """
        Reset the simulated world
        """
        self.pause_sim()
        self.initialize_sim()

    def save_map(self, filename):
        """
        Save the map
        :param filename: Filename under which the map shall be stored
        """
        self.map_manager.save_map(filename)

    def load_map(self, filename):
        """

        :param filename:
        """
        self.map_manager.load_map(filename)
        self.reset_sim()

    def random_map(self):
        self.pause_sim()
        self.initialize_sim(random=True)

    def reset_grid_map(self):
        for robot in self.world.robots:
            robot.supervisor.reset_grid_mapping()

    def draw_world(self):
        self.viewer.new_frame()  # start a fresh frame
        self.world_plotter.draw_world_to_frame()  # draw the world onto the frame
        # draw occupancy grid maps to the frame
        if self.viewer.start_mapping:
            for plotter in self.reg_mapping_plotters:
                if plotter is not None:
                    plotter.draw_mapping_to_frame()
        # draw the slam estimations to the frame
        for plotter in self.reg_slam_plotters:
            if plotter is not None:
                plotter.draw_slam_to_frame()

        self.viewer.draw_frame()  # render the frame

    def _run_sim(self):
        self.sim_event_source = GLib.timeout_add(int(self.period * 1000), self._run_sim)
        self._step_sim()

    def _update_slam_accuracies(self):
        # Only perform the SLAM evaluation on specific simulation cycles. The period is configurable.
        if self.num_cycles % self.cfg["slam"]["evaluation"]["interval"] == 0:
            if self.ekfslam_evaluation is not None:
                self.ekfslam_evaluation.evaluate(self.world.obstacles)
            if self.fastslam_evaluation is not None:
                self.fastslam_evaluation.evaluate(self.world.obstacles)
            if self.graphbasedslam_evaluation is not None:
                self.graphbasedslam_evaluation.evaluate(self.world.obstacles)

    def _step_sim(self):
        self.num_cycles += 1
        # increment the simulation
        try:
            self.world.step()
        except CollisionException:
            self.end_sim('Collision!')
        except GoalReachedException:
            if self.cfg["map"]["goal"]["endless"]:
                self.map_manager.add_new_goal()
                self.map_manager.apply_to_world(self.world)
            else:
                self.end_sim("Goal Reached!")

        # Evaluate accuracies of slam
        self._update_slam_accuracies()

        # draw the resulting world
        self.draw_world()


if __name__ == "__main__":
    filename = "config.yaml" if len(sys.argv) == 1 else sys.argv[1]
    with open(filename, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    Simulator(cfg)
