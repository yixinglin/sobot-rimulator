class MappingPlotter:

    def __init__(self, slam_mapping, viewer, frame_number):
        """
        Initialize a MappingPlotter object
        :param slam_mapping: An object of OccupancyMapping2d
        :param viewer: The viewer to be used
        :param frame_number: The frame number to be used
        """
        self.slam_mapping = slam_mapping
        self.viewer = viewer
        self.frame_number = frame_number
        self.resolution = slam_mapping.resolution # number of pixels per meter
        self.pixel_size = 1/self.resolution # pixel size in meters
        self.offset = slam_mapping.offset  # offset in meters

    def draw_mapping_to_frame(self):
        """
        Draw the occupancy mapping estimated by the mapping algorithm to the frame
        """
        if self.viewer.draw_invisibles == True:
            frame = self.viewer.current_frames[self.frame_number]
            H, W = self.slam_mapping.map_shape()
            map = self.slam_mapping.get_map()
            for j in range(H):
                for i in range(W):
                    val = map[j, i]  # the occupancy probability
                    if val >= 0.5:
                        x, y = self.__to_meter(i, j)
                        frame.add_rectangle([x, y], self.pixel_size, self.pixel_size, color=(0.5, 0.5, 0.5), alpha=0.5)
            self.draw_path_planning_to_frame(frame)
            self._draw_goal_to_frame(frame)

    def draw_path_planning_to_frame(self, frame):
        """
        Draw the path to the frame
        :param frame: The frame to be used
        """
        if self.viewer.draw_invisibles == True:
            path = self.slam_mapping.get_path()
            if len(path) > 1:
                frame.add_lines([path],
                    linewidth=0.010,
                    color="red1",
                    alpha=0.9)

    def _draw_goal_to_frame(self, frame):
        """
        Draw the current goal to the frame
        :param frame: The frame to be used
        """
        goal = self.slam_mapping.supervisor.goal()
        frame.add_circle(pos=goal,
                         radius=0.05,
                         color="dark green",
                         alpha=0.65)
        frame.add_circle(pos=goal,
                         radius=0.01,
                         color="black",
                         alpha=0.5)

    def __to_meter(self, x, y):
        """
        Calculate the position of a point in pixels by its position in meters
        :param x: x in pixels
        :param y: y in pixels
        :return:
                x, y: the position of a point in meters
        """
        x = x/self.resolution - self.offset[0]
        y = y/self.resolution - self.offset[1]
        return x, y