from math import sqrt
from matplotlib import pyplot as plt

from supervisor.slam.EKFSlam import EKFSlam
from supervisor.slam.FastSlam import FastSlam
from supervisor.slam.GraphBasedSLAM import GraphBasedSLAM
from models.obstacles.FeaturePoint import FeaturePoint

class SlamEvaluation:
    def __init__(self, slam, evaluation_cfg, robot):
        """
        Initializes an object of the SlamEvaluation class
        :param slam: The slam algorithm that will be evaluated
        :param evaluation_cfg: The configurations for the class.
                               Currently only used to calculate number of simulation cycles
        :param robot: robot object in the real world
        """
        self.slam = slam
        self.cfg = evaluation_cfg
        self.average_distances_lm = []
        self.distances_robot = []
        self.robot = robot

    def evaluate(self, obstacles):
        """
        Evaluates the average distance of the estimated obstacle positions to the closest actual obstacle in the map.
        The value is saved.
        :param obstacles: The list of actual obstacles of the map
        """
        slam_obstacles = self.slam.get_landmarks()
        squared_distances = []
        for i, slam_obstacle in enumerate(slam_obstacles):
            slam_obstacle_id = slam_obstacle[2]
            for obstacle in obstacles:
                if type(obstacle) == FeaturePoint and obstacle.id == slam_obstacle_id:
                    sq_dist = self.__calc_squared_distance(slam_obstacle[:2], obstacle.pose.sunpack())
                    squared_distances.append(sq_dist)
        self.average_distances_lm.append(sum(squared_distances) / len(squared_distances))

        slam_pose = self.slam.get_estimated_pose()
        self.distances_robot.append(self.__calc_squared_distance(slam_pose.sunpack(), self.robot.pose.sunpack()))

    def plot(self):
        """
        Produces a plot of how the average distance changed over the course of the simulation.
        Saves the plot in a png file.
        """
        fig, ax = plt.subplots()
        # Calculates number of elapsed simulation cycles
        sim_cycles = len(self.average_distances_lm) * self.cfg["interval"]
        ax.plot(range(0, sim_cycles, self.cfg["interval"]), self.average_distances_lm)
        ax.plot(range(0, sim_cycles, self.cfg["interval"]), self.distances_robot)
        ax.grid()
        if isinstance(self.slam, EKFSlam):
            ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark in meters',
                   title='Evaluation of EKF SLAM')
            plt.savefig('ekf_slam_evaluation.png')
        elif isinstance(self.slam, FastSlam):
            ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark in meters',
                   title='Evaluation of FastSLAM')
            plt.savefig('fast_slam_evaluation.png')
        elif isinstance(self.slam, GraphBasedSLAM):
            ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark in meters',
                   title='Evaluation of Graph-based Slam')
            plt.savefig('graph_based_slam_evaluation.png')

        ax.grid()

        plt.show()

    def __find_min_distance(self, slam_obstacle, obstacles):
        """
        Finds the distance of the estimated obstacle to the the closest actual obstacle
        :param slam_obstacle: An estimated obstacle position of a SLAM algorithm
        :param obstacles: The list of actual obstacles in the map
        :return: Distance of estimated obstacle to closest actual obstacle
        """
        squared_distances = [self.__calc_squared_distance(slam_obstacle, obstacle.pose.sunpack()) for obstacle in obstacles]
        return sqrt(min(squared_distances))

    @staticmethod
    def __calc_squared_distance(x, y):
        """
        Calculates squared distance between two positions.
        The squared distance is sufficient for finding the minimum distance.
        :param x: First position
        :param y: Second position
        :return: squared distance between the two positions
        """
        diff = (x[0] - y[0], x[1] - y[1])
        return diff[0] ** 2 + diff[1] ** 2
