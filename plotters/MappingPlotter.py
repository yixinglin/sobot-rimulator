from models.obstacles.RectangleObstacle import RectangleObstacle
from plotters.ObstaclePlotter import ObstaclePlotter
from models.Pose import Pose
class MappingPlotter:

    def __init__(self, slam_mapping, viewer, frame_number):
        """

        :param slam_mapping: An object of OccupancyGridMap2d
        :param viewer: The viewer to be used
        :param frame_number: The frame number to be used
        """
        self.slam_mapping = slam_mapping
        self.viewer = viewer
        self.frame_number = frame_number

        self.resolution = slam_mapping.resolution # number of pixels per meter
        self.pixel_size = 1/self.resolution # pixel size in meters
        self.offset = slam_mapping.offset  # offset in meters
        self.counter = 0

    def draw_mapping_to_frame(self):
        frame = self.viewer.current_frames[self.frame_number]
        H, W = self.slam_mapping.map_shape()
        for j in range(H):
            for i in range(W):
                val = self.slam_mapping.get_value(i, j)
                if val > 0.5:
                    x, y = self.__to_meter(i, j)
                    frame.add_circle([y, x], self.pixel_size/2, color='gray73', alpha=val)

    def __to_meter(self, x, y):
        x = x/self.resolution - self.offset[0]
        y = y/self.resolution - self.offset[1]
        return x, y




