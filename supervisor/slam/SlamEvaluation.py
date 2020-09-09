from math import sqrt
from matplotlib import pyplot as plt

from supervisor.slam.EKFSlam import EKFSlam
from supervisor.slam.FastSlam import FastSlam
from supervisor.slam.GraphBasedSLAM import GraphBasedSLAM

class SlamEvaluation:
    def __init__(self, slam, evaluation_cfg):
        """
        Initializes an object of the SlamEvaluation class
        :param slam: The slam algorithm that will be evaluated
        :param evaluation_cfg: The configurations for the class.
                               Currently only used to calculate number of simulation cycles
        """
        self.slam = slam
        self.cfg = evaluation_cfg
        self.average_distances = []

    def evaluate(self, obstacles):
        """
        Evaluates the average distance of the estimated obstacle positions to the closest actual obstacle in the map.
        The value is saved.
        :param obstacles: The list of actual obstacles of the map
        """
        slam_obstacles = self.slam.get_landmarks()
        min_distances = [self.__find_min_distance(slam_obstacle, obstacles) for slam_obstacle in slam_obstacles]
        self.average_distances.append(sum(min_distances) / len(min_distances))

    def plot(self):
        """
        Produces a plot of how the average distance changed over the course of the simulation.
        Saves the plot in a png file.
        """
        fig, ax = plt.subplots()
        # Calculates number of elapsed simulation cycles
        sim_cycles = len(self.average_distances) * self.cfg["interval"]
        ax.plot(range(0, sim_cycles, self.cfg["interval"]), self.average_distances)
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
