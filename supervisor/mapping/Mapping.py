
class Mapping:
    """
    An abstract class for mapping algorithms
    """
    def update(self, z):
        """
        Updates the map and the path using mapping and path planning algorithms
        :param z: Measurements data from sensors
        """
        raise NotImplementedError()

    def get_map(self):
        """
        Returns the estimated occupancy grid map
        """
        raise NotImplementedError()

    def get_path(self):
        """
        Returns the estimated path from a path planner
        """
        raise NotImplementedError()



