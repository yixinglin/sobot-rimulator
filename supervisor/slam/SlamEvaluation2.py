from math import sqrt
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from models.obstacles.FeaturePoint import FeaturePoint
import pickle
import pandas as pd

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
    self.reset_record()
    self.sim_circle = 0

  def reset_record(self):
    self.head1 = ["sim_circle", "landmark_id", "estimated_landmark_position", "estimated_robot_pose",
                 "actual_landmark_position", "actual_robot_pose", "slam_name"]
    self.data = list()

    self.head2 = ["sim_circle", "name", "time_per_update"]
    self.update_time_record = list()

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

  def record(self, sim_circle, obstacles):
    """
    Record data in a list
    :param sim_circle: simulation circle
    """
    self.sim_circle = sim_circle
    for j, slam in enumerate(self.list_slam):
      if slam is None:
        continue
      slam_landmarks = slam.get_landmarks()
      estimated_robot_position = slam.get_estimated_pose().sunpack()
      actual_robot_position = self.robot.pose.sunpack()
      slam_name = str(slam)
      for i, slam_landmark in enumerate(slam_landmarks):
        landmark_id = slam_landmark[2]
        estimated_landmark_position = slam_landmark[:2]
        actual_landmark_position = None

        for obstacle in obstacles:
          if type(obstacle) == FeaturePoint and obstacle.id == landmark_id:
            actual_landmark_position = obstacle.pose.sunpack()
            break

        self.data.append([int(sim_circle), int(landmark_id), estimated_landmark_position,
                estimated_robot_position, actual_landmark_position, actual_robot_position, slam_name])

    if sim_circle % 1000 == 0:
      print ("sim_circle", sim_circle)

  def time_per_step(self, name, time):
    """
    A callback function for recording time used per update step.
    :param name: Name of slam or mapping algorithm
    :param time: time cost by updating
    """
    if self.sim_circle > 0:
      self.update_time_record.append((self.sim_circle, name, time))

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
    self.save_infomation("./scripts/sobot_information")

    fig, ax = plt.subplots(2, figsize=(9,8))

    ax[0].grid()
    line_styles = ['k-', 'k--', 'k:',  'k-.']
    for i, slam in enumerate(self.list_slam):
      if slam is None:
        continue

      name = str(slam)
      # Calculates number of elapsed simulation cycles
      sim_cycles = len(self.average_distances_lm[i]) * self.cfg["interval"]
      ax[0].plot(range(0, sim_cycles, self.cfg["interval"]),
              self.average_distances_lm[i],
              line_styles[i%len(self.list_slam)], label = name)
    ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    ax[0].legend()
    ax[0].set(xlabel='Simulation cycles', ylabel='Average distance to true landmark in meters', title='Evaluation of SLAM')
    plt.savefig('slam_landmark_evaluation.png')
    ax[0].grid()
    plt.show()


    #fig, ax = plt.subplots()
    ax[1].grid()
    for i, slam in enumerate(self.list_slam):
      if slam is None:
        continue

      name = str(slam)
      # Calculates number of elapsed simulation cycles
      sim_cycles = len(self.average_distances_lm[i]) * self.cfg["interval"]
      ax[1].plot(range(0, sim_cycles, self.cfg["interval"]), self.distances_robot[i], line_styles[i%len(self.list_slam)], label = name)
    ax[1].legend()
    ax[1].set(xlabel='Simulation cycles', ylabel='Distance to true robot position in meters', title='Evaluation of SLAM')
    plt.savefig('slam_robot_position_evaluation.png')
    ax[1].grid()
    plt.tight_layout()
    plt.show()

  def save_infomation(self, filename):
      """
      Save the information
      :param filename: The filename under which the information shall be stored
      """
      df = pd.DataFrame(columns=self.head1, data=self.data)
      df.to_csv(filename + "1.csv")
      df = pd.DataFrame(columns=self.head2, data=self.update_time_record)
      df.to_csv(filename + "2.csv")

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