from math import sqrt
from matplotlib import pyplot as plt


class SlamEvaluation:
    def __init__(self, slam, ekf=True):
        self.slam = slam
        self.ekf = ekf
        self.average_distances = []

    def step(self, obstacles):
        slam_obstacles = self.slam.get_landmarks()
        min_distances = [self._find_min_distance(slam_obstacle, obstacles) for slam_obstacle in slam_obstacles]
        self.average_distances.append(sum(min_distances) / len(min_distances))

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(range(len(self.average_distances)), self.average_distances)
        if self.ekf:
            ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark',
                   title='Evaluation of EKF SLAM')
        else:
            ax.set(xlabel='Simulation cycles', ylabel='Average distance to true landmark',
                   title='Evaluation of FastSLAM')
        ax.grid()

        plt.show()


    def _find_min_distance(self, slam_obstacle, obstacles):
        distances = [self._calc_distance(slam_obstacle, obstacle.pose.sunpack()) for obstacle in obstacles]
        return min(distances)

    def _calc_distance(self, x, y):
        diff = (x[0] - y[0], x[1] - y[1])
        return sqrt(diff[0] ** 2 + diff[1] ** 2)
