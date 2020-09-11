import numpy as np
from math import cos, sin, sqrt
from utils.geometrics_util import bresenham_line
class OccupancyGridMap2d:

    def __init__ (self, slam, slam_cfg, max_range):
        """
        Initialize the OccupancyGridMap2d object
        :param slam: The underlying slam algorithm object.
        :param slam_cfg: The slam configuration.
        """
        self.slam = slam
        self.width = slam_cfg['mapping']['gridmap']['width'] # width of the map in meters
        self.height = slam_cfg['mapping']['gridmap']['height'] # height of the map in meters
        self.resolution = slam_cfg['mapping']['gridmap']['resolution'] # resolution of the map, i.e. number of pixels per meter
        self.W = int(self.width * self.resolution) # width of the map in pixels
        self.H = int(self.height * self.resolution) # height of the map in pixels
        self.offset = (slam_cfg['mapping']['gridmap']['offset']['x'],  # offset in meters in horizontal direction
                        slam_cfg['mapping']['gridmap']['offset']['y']) # offset in meters in vertical directionss
        self.max_range = max_range
        self.map = np.zeros((self.H, self.W), dtype=np.float32) # a map where each pixel represents a probability the corresponding grid is occupied.

        self.prob_unknown = 0.5 # prior
        self.prob_occ = 0.4 # probability that a grid is occupied
        self.L = self.__prob2log(np.ones_like(self.map) * self.prob_unknown)  # recursive term of the map
        self.L0 = self.__prob2log(np.ones_like(self.map) * self.prob_unknown)  # prior term of the map

    def update(self, z):
        """
        Update the occupancy gridmap recursively
        :param z: Measurement, represented as tuple of measured distance and measured angle
        """
        observed_pixs = []
        lines = self.__calc_lines(z)
        for x0, y0, x1, y1 in lines:
            x0, y0 = self.__to_gridmap_position(x0, y0) # from position in meters to position in pixels
            x1, y1 = self.__to_gridmap_position(x1, y1) # from position in meters to position in pixels
            points = bresenham_line(x0, y0, x1, y1) # a list of points on the line
            points = self.__calc_prob(points)  # calculate the occupancy probabilities on this line
            observed_pixs += points

        inverse_sensor = np.ones_like(self.map) * self.prob_unknown # initialize the inverse-sensor-model term by prior

        for xi, yi, prob_occ in observed_pixs:
            if xi < self.W and xi >= 0 and yi < self.H and yi >= 0:
                inverse_sensor[yi, xi] = prob_occ
        self.L = self.L + self.__prob2log(inverse_sensor) - self.L0 # update the recursive term

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
                prob = 1-self.prob_occ
            probs.append((xi, yi, prob))
        return probs

    def get_map(self):
        """
        :return: a 2D numpy.array of the occupied gridmap.
                    A single value represents the probability of the occupancy
        """
        map = self.__log2prob(self.L)
        return map

    def map_shape(self):
        """
        Get map shape
        :return: a tuple of shape, i.e. (rows, columns)
        """
        return self.map.shape

    def blur(self):
        """
        Blur the map
        :return:
        """
        pass

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
        :param x: x in meter
        :param y: y in meter
        :return:
                pix_x in pixel
                pix_y in pixel
        """
        pix_x = (x + self.offset[0]) * self.resolution
        pix_y = (y + self.offset[1]) * self.resolution
        return int(pix_x), int(pix_y)

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