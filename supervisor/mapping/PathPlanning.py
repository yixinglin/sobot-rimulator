
class PathPlanning:
    """
    An abstract class for path planning algorithm
    """
    def execute(self, sx, sy, gx, gy, obstacle_map):
        """
        path searching
        :param sx: x coordinate of the starting position.
        :param sy: y coordinate of the starting position.
        :param gx: x coordinate of the goal position.
        :param gy: y coordinate of the goal position.
        :param obstacle_map: a binary 2d ndarray. The value of a iterm is Ture (obstacle) or False (free).
        :return:
                shortest_path: a list of coordinates (x, y) along the path
        """
        raise NotImplementedError()

