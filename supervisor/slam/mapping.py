"""
This file includes
- occupancy mapping algorithm
- path planning algorithm

"""

import numpy as np
import math
from scipy.ndimage import gaussian_filter
from math import cos, sin, sqrt
from utils.geometrics_util import bresenham_line

class Mapping:
    """
    An abstract class for a mapping algorithm
    """
    def update(self, z):
        """
        Update the map and the path using mapping and path planning algorithms
        :param z: Measurements data from sensors
        """
        raise NotImplementedError()

    def get_map(self):
        """
        Returns the estimated occupancy map
        """
        raise NotImplementedError()

    def get_path(self):
        """
        Returns the estimated path of planning
        """
        raise NotImplementedError()



class OccupancyMapping2d(Mapping):
    def __init__ (self, slam, slam_cfg, supervisor_interface, path_planner = None):
        """
        Initialize the OccupancyMapping2d object
        :param slam: The underlying slam algorithm object.
        :param slam_cfg: The slam configuration.
        :param path_planner: An object of PathPlanner
        """
        self.supervisor = supervisor_interface
        self.slam = slam
        self.path_planner = path_planner
        self.width = slam_cfg['mapping']['gridmap']['width'] # width of the map in meters
        self.height = slam_cfg['mapping']['gridmap']['height'] # height of the map in meters
        self.resolution = slam_cfg['mapping']['gridmap']['resolution'] # resolution of the map, i.e. number of pixels per meter
        self.W = int(self.width * self.resolution) # width of the map in pixels
        self.H = int(self.height * self.resolution) # height of the map in pixels
        self.offset = (slam_cfg['mapping']['gridmap']['offset']['x'],  # offset in meters in horizontal direction
                        slam_cfg['mapping']['gridmap']['offset']['y']) # offset in meters in vertical directionss
        self.max_range = supervisor_interface.proximity_sensor_max_range()

        self.prob_unknown = 0.5 # prior
        self.prob_occ = 0.99 # probability perceptual a grid is occupied
        self.prob_free = 0.01 # probability perceptual a grid is free
        self.map = np.full((self.H, self.W), self.prob_unknown, dtype=np.float32)
        self.L = self.__prob2log(np.full_like(self.map, self.prob_unknown, dtype=np.float32))  # log recursive term of the map
        self.L0 = self.__prob2log(np.full_like(self.map, self.prob_unknown, dtype=np.float32))  # log prior term of the map

        self.path = list() # a path calculated by path planner
        self.update_counter = 0

    def update(self, z):
        """
        Update the occupancy gridmap recursively
        Update the path planning
        :param z: Measurement, represented as tuple of measured distance and measured angle
        """
        observed_pixs = [] # a list of grid positions and its occupancy probabilities
        lines = self.__calc_lines(z)
        for x0, y0, x1, y1 in lines:
            x0, y0 = self.__to_gridmap_position(x0, y0) # from position in meters to position in pixels
            x1, y1 = self.__to_gridmap_position(x1, y1) # from position in meters to position in pixels
            points = bresenham_line(x0, y0, x1, y1) # a list of points on the line
            occ_probs = self.__calc_prob(points)  # calculate the occupancy probabilities on this line
            observed_pixs += occ_probs

        inverse_sensor = np.copy(self.L0) # initialize the inverse-sensor-model term by prior

        for xi, yi, prob_occ in observed_pixs:
            if xi < self.W and xi >= 0 and yi < self.H and yi >= 0:
                inverse_sensor[yi, xi] = math.log((prob_occ/(1-prob_occ)))
        self.L = self.L + inverse_sensor - self.L0  # update the recursive term
        self.L = np.clip(self.L, -5, 5)
        self.update_counter += 1
        self.__update_path_planning()      # update path planning

    def get_path(self):
        """
        Get the path from the path planner.
        :return: A list of points on the path. A single item is (x, y) in meters.
        """
        if len(self.path) > 1:
            world_path = [self.__to_world_position(xi, yi) for xi, yi in self.path]
        else:
            world_path = list()
        return world_path

    def get_map(self):
        """
        :return: a 2D numpy.array of the occupied gridmap.
                    A single value represents the probability of the occupancy
        """
        self.map = self.__log2prob(self.L)
        return self.map

    def map_shape(self):
        """
        Get map shape
        :return: a tuple of shape, i.e. (rows, columns)
        """
        return self.map.shape

    def __update_path_planning(self):
        occ_threshold = 0.2
        if self.path_planner is not None and self.update_counter % 5 ==0 and self.update_counter > 5:
            goal = self.supervisor.goal()  # get the goal
            start = self.slam.get_estimated_pose().sunpack() # get the estimated pose from slam
            gx, gy = self.__to_gridmap_position(goal[0], goal[1])
            sx, sy = self.__to_gridmap_position(start[0], start[1])
            self.map[sy, sx] = 0
            if self.map[gy, gx] < occ_threshold:
                bool_map = self.blur(self.map)
               # bool_map = np.copy(self.map)  # calculate a boolean map
                bool_map[bool_map >= occ_threshold] = True
                bool_map[bool_map < occ_threshold] = False
                bool_map = bool_map.astype(np.bool)
                bool_map[sy, sx] = False
                bool_map[gy, gx] = False
                self.path = self.path_planner.planning(sx, sy, gx, gy, bool_map, type='euclidean')
            else:
                self.path = list()

    def __calc_prob(self, points):
        """
        Calculate the probability the occupied points on a segment
        :param points: points on a segment in pixels
        :return:  A list of occupancy probabilities of the grids on the segment.
                    A single item is (x, y, prob) where (x, y) is the prosition of a grid, and prob is the occupancy probability.
        """
        x0, y0 = points[0] # start point position
        xn, yn = points[-1] # end point position
        seg_length = sqrt((xn - x0)**2 + (yn - y0)**2) # the length of the segment
        max_range_seg =  self.max_range*self.resolution
        probs = [] # occupancy probabilities of the grids on the segment.
        for xi, yi in points:
            prob = self.prob_unknown
            if seg_length >= max_range_seg:     # out of range
                prob = self.prob_unknown        # assign prob with prior
            if seg_length < max_range_seg:    # likely to be an object
                prob = self.prob_occ            # assign prob with the occupied probability

            distance = sqrt((xi - x0)**2 + (yi - y0)**2)
            if distance < seg_length:                  # free space
                prob = self.prob_free
            probs.append((int(xi), int(yi), prob))
        return probs

    def blur(self, image):
        """
        Blur the map
        :return:
        """
        return gaussian_filter(image, sigma=1)

    def __calc_lines(self, z):
        """
        Calculate the start points end end points of segments.
        :param z: Measurement, represented as tuple of measured distance and measured angle
        :return:
                A list of segments. A single line is represented by (x0, y0, x1, y1)
                    where (x0, y0) is the position of the start point and (x1, y1) is the position of the end point.
        """
        pose = self.slam.get_estimated_pose().sunpack()
        lines = []
        for measurement in z:
            lmx, lmy = self.calc_landmark_position(pose, measurement)
            lines.append((pose[0], pose[1], lmx, lmy))
        return lines

    def __to_gridmap_position(self, x, y):
        """
        Calculate the position of a pixel in the grid map
        :param x: x in meters
        :param y: y in meters
        :return:
                pix_x in pixels
                pix_y in pixels
        """
        pix_x = round((x + self.offset[0]) * self.resolution)
        pix_y = round((y + self.offset[1]) * self.resolution)
        return int(pix_x), int(pix_y)

    def __to_world_position(self, x, y):
        """
        Calculate the position of a pixel in the grid map
        :param x: x in pixels
        :param y: y in pixels
        :return:
                pix_x in meters
                pix_y in meters
        """
        meter_x = x / self.resolution - self.offset[0]
        meter_y = y / self.resolution - self.offset[1]
        return meter_x, meter_y

    def __prob2log(self, p):
        """
        :param p: probability that a grid is occupied
        :return: log likelihood
        """
        return np.log(p/(1-p))

    def __log2prob(self, lx):
        """
        :param lx: log likelihood
        :return: probability that a grid is occupied
        """
        return 1 - 1/(1+np.exp(lx))

    @staticmethod
    def calc_landmark_position(x, z):
        """
        Returns the measured landmark position
        :param x: The robots pose (or combined state vector, only matters that first three elements are robot pose)
        :param z: Measurement, represented as tuple of measured distance and measured angle
        :return: Measured landmark position
        """
        lm = [0, 0]
        lm[0] = x[0] + z[0] * cos(z[1] + x[2])
        lm[1] = x[1] + z[0] * sin(z[1] + x[2])
        return lm