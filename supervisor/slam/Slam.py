
"""
An abstract class for a feature-based SLAM algorithm
"""
class Slam:

    def get_estimated_pose(self):
        raise NotImplementedError()

    def get_landmarks(self):
        raise NotImplementedError()

    def execute(self, u, z):
        """
        Executes an update cycle of the SLAM algorithm
        :param u: motion command
        :param z: List of measurements
        """
        raise NotImplementedError()



