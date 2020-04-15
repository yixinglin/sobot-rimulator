
class Slam:
    """
    An abstract class for a feature-based SLAM algorithm
    """

    def get_estimated_pose(self):
        """
        Returns the estimated pose of the robot
        """
        raise NotImplementedError()

    def get_landmarks(self):
        """
        Returns the estimated landmark positions
        """
        raise NotImplementedError()

    def update(self, u, z):
        """
        Executes an update cycle of the SLAM algorithm
        :param u: motion command
        :param z: List of measurements
        """
        raise NotImplementedError()



