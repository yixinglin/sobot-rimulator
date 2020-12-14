import numpy as np

class MappingPlotter:

    def __init__(self, slam_mapping, viewer, frame_number):
        """
        Initialize a MappingPlotter object
        :param slam_mapping: An object of OccupancyGridMapping2d
        :param viewer: The viewer to be used
        :param frame_number: The frame number to be used
        """
        self.slam_mapping = slam_mapping
        self.viewer = viewer
        self.frame_number = frame_number
        self.grid_resolution = slam_mapping.resolution # number of grids per meter
        self.offset = slam_mapping.offset  # offset in meters
        self.pixels_per_meter = viewer.pixels_per_meter
        self.alpha = 200 # alpha value of color

    def draw_mapping_to_frame(self):
        """
        Draw the occupancy mapping estimated by the mapping algorithm to the frame
        """
        self.slam_mapping.update_enabled = True
        frame = self.viewer.current_frames[self.frame_number]
        map = self.slam_mapping.get_map()
        map = 255 - map * 255
        map.astype(int)

        image = self.matrix_to_image(map, self.pixels_per_meter/self.grid_resolution, self.alpha)
        tx, ty = self.offset
        frame.add_background_image(image, (tx*self.pixels_per_meter, ty*self.pixels_per_meter))

        self._draw_goal_to_frame(frame)

        self.draw_path_planning_to_frame(frame)

    def draw_path_planning_to_frame(self, frame):
        """
        Draw the path to the frame
        :param frame: The frame to be used
        """

        path = self.slam_mapping.get_path()
        if len(path) > 1:
            frame.add_lines([path],
                linewidth=0.010,
                color="red1",
                alpha=0.9)

    @staticmethod
    def matrix_to_image(m, resolution, alpha):
        """
        Convert the grid map to an image
        :param m:  numpy array in size of H x W with value from 0 - 255
        :param resolution: pixels per grid
        :param alpha: Alpha value of the color
        :return: an image of map in RGBA format
        """
        H, W = m.shape
        #resolution = resolution
        img = np.zeros(( round(H * resolution),
                         round(W * resolution), 4))
        img[:, :, 3] = alpha
        for j in range(0, H):
            for i in range(0, W):
                ii1 = round(i * resolution)
                ii2 = round((i+1) * resolution)
                jj1 = round(j * resolution)
                jj2 = round((j+1) * resolution)
                img[jj1:jj2, ii1:ii2, 0:3] = m[j, i]
        return img

    def pixbuf_from_array(z):
        " Convert from numpy array to GdkPixbuf "
        z = z.astype('uint8')
        h, w, c = z.shape
        assert c == 3 or c == 4
        if hasattr(GdkPixbuf.Pixbuf, 'new_from_bytes'):
            Z = GLib.Bytes.new(z.tobytes())
            return GdkPixbuf.Pixbuf.new_from_bytes(Z, GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w * c)
        return GdkPixbuf.Pixbuf.new_from_data(z.tobytes(), GdkPixbuf.Colorspace.RGB, c == 4, 8, w, h, w * c, None, None)

    def _draw_goal_to_frame(self, frame):
        """
        Draw the current goal to the frame
        :param frame: The frame to be used
        """
        goal = self.slam_mapping.supervisor.goal()
        frame.add_circle(pos=goal,
                         radius=0.05,
                         color="dark red",
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
        x = x/self.grid_resolution - self.offset[0]
        y = y/self.grid_resolution - self.offset[1]
        return x, y