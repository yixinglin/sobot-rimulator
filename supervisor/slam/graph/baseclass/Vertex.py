"""
The parent class of any types of vertex
"""

import numpy as np
class Vertex:

    def __init__(self, pose, sigma, observation = None):
        """
        A vertex class. It is a component of a graph
        :param pose: a vector of pose
        :param sigma: a covariance matrix, uncertainty of measurement
        :param observation:  sensor observation
        """
        self.id = -1  # identifier of a vertex
        self.pose = np.copy(pose)
        self.observation = np.copy(observation)
        self.sigma = np.copy(sigma)
        self.dim = pose.shape[0]  # dimension of the pose vector  # vector dimension.

    def update_pose(self, pose):
        self.pose = pose




