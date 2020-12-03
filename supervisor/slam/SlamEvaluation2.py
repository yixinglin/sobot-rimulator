from math import sqrt
from matplotlib import pyplot as plt

from supervisor.slam.EKFSlam import EKFSlam
from supervisor.slam.FastSlam import FastSlam
from supervisor.slam.GraphBasedSLAM import GraphBasedSLAM
from models.obstacles.FeaturePoint import FeaturePoint


class SlamEvaluation2:
  def __init__(self, list_slam, evaluation_cfg, robot):
    """
    Initializes an object of the SlamEvaluation2 class
    :param list_slam: a list of slam algorithms that will be evaluated
    :param evaluation_cfg: The configurations for the class.
                           Currently only used to calculate number of simulation cycles
    :param robot: robot object in the real world
    """
    self.list_slam = list_slam
    self.cfg = evaluation_cfg
    self.average_distances_lm = [list() for _ in list_slam]
    self.distances_robot = [list() for _ in list_slam]
    self.robot = robot

  def evaluate(self, obstacles):
    """
    Evaluates the average distance of the estimated obstacle positions to the closest actual obstacle in the map.
    The value is saved.
    :param obstacles: The list of actual obstacles of the map
    """
    for j, slam in enumerate(self.list_slam):
      if slam is None:
        continue
      slam_obstacles = slam.get_landmarks()
      distances = []
      for i, slam_obstacle in enumerate(slam_obstacles):
        if self.cfg["associate_by_id"] == True:
          dist = self.__find_distance_by_landmark_id(slam_obstacle, obstacles)
        else:
          dist = self.__find_min_distance(slam_obstacle, obstacles)
        distances.append(dist)

      if (len(distances)) > 0:
        self.average_distances_lm[j].append(sum(distances) / len(distances))
        slam_pose = slam.get_estimated_pose()
        self.distances_robot[j].append(self.__calc_squared_distance(slam_pose.sunpack(), self.robot.pose.sunpack()))

  def __find_distance_by_landmark_id(self, slam_obstacle, obstacles):
    """
    Finds the distance of the estimated obstacle to the actual obstacle regarding its identifiers
    :param slam_obstacle: An estimated obstacle position of a SLAM algorithm
    :param obstacles: The list of actual obstacles in the map
    :return: Distance of estimated obstacle to the actual obstacle
    """
    slam_obstacle_id = slam_obstacle[2]
    for obstacle in obstacles:
      if type(obstacle) == FeaturePoint and obstacle.id == slam_obstacle_id:
        sq_dist = self.__calc_squared_distance(slam_obstacle[:2], obstacle.pose.sunpack())
        return sqrt(sq_dist)

  def __find_min_distance(self, slam_obstacle, obstacles):
      """
      Finds the distance of the estimated obstacle to the the closest actual obstacle
      :param slam_obstacle: An estimated obstacle position of a SLAM algorithm
      :param obstacles: The list of actual obstacles in the map
      :return: Distance of estimated obstacle to closest actual obstacle
      """
      squared_distances = [self.__calc_squared_distance(slam_obstacle, obstacle.pose.sunpack()) for obstacle in obstacles]
      return sqrt(min(squared_distances))

  def plot(self):
    """
    Produces a plot of how the average distance changed over the course of the simulation.
    Saves the plot in a png file.
    """
    fig, ax = plt.subplots()
    ax.grid()
    line_styles = ['k-', 'k--', 'k:',  'k-.']
    for i, slam in enumerate(self.list_slam):
      if slam is None:
        continue
      if isinstance(slam, EKFSlam):
        name = "EKF SLAM"
      elif isinstance(slam, FastSlam):
        name = "FastSLAM"
      elif isinstance(slam, GraphBasedSLAM):
        name = "Graph-based SLAM"
      else:
        name = "SLAM"
      # Calculates number of elapsed simulation cycles
      sim_cycles = len(self.average_distances_lm[i]) * self.cfg["interval"]
      ax.plot(range(0, sim_cycles, self.cfg["interval"]), self.average_distances_lm[i], line_styles[i%len(self.list_slam)], label = name)

    ax.legend()
    ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark in meters', title='Evaluation of SLAM')
    plt.savefig('slam_landmark_evaluation.png')
    ax.grid()
    plt.show()


    fig, ax = plt.subplots()
    ax.grid()
    for i, slam in enumerate(self.list_slam):
      if slam is None:
        continue
      if isinstance(slam, EKFSlam):
        name = "EKF SLAM"
      elif isinstance(slam, FastSlam):
        name = "FastSLAM"
      elif isinstance(slam, GraphBasedSLAM):
        name = "Graph-based SLAM"
      else:
        name = "SLAM"
      # Calculates number of elapsed simulation cycles
      sim_cycles = len(self.average_distances_lm[i]) * self.cfg["interval"]
      ax.plot(range(0, sim_cycles, self.cfg["interval"]), self.distances_robot[i], line_styles[i%len(self.list_slam)], label = name)
    ax.legend()
    ax.set(xlabel='Simulation cycles', ylabel='Distance to true robot position in meters', title='Evaluation of SLAM')
    plt.savefig('slam_robot_position_evaluation.png')
    ax.grid()
    plt.show()


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