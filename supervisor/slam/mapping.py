import numpy as np
class OccupancyGridMap2d:

    def __init__ (self, slam, supervisor_interface, slam_cfg):
        """

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
                        slam_cfg['mapping']['gridmap']['offset']['y']) # offset in meters in vertical direction

        self.map = np.zeros((self.H, self.W), dtype=np.float32) # a map where each pixel represents a probability the corresponding grid is occupied.

    def update(self):
        x, y = self.__to_gridmap_position(1.5,1.5)
        self.map[y, x] = 0.5
        x, y = self.__to_gridmap_position(1.5, 2)
        self.map[y, x] = 0.8
        # print ("update mapping")

    def get_map(self):
        return self.map

    def get_occupied_positions(self):
        indices = np.argwhere(self.map > 0.5)
        y, x = zip(*indices)
        return y, x

    def get_value(self, x, y):
        return self.map[y, x]

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


