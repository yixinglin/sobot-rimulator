"""
The parent class of any types of vertex
"""

import numpy as np
class Vertex:

    def __init__(self, pose, sigma):
        """
        A vertex class. It is a component of a graph
        :param pose: a vector of pose
        :param sigma: a covariance matrix, uncertainty of measurement
        """
        # identifier of a vertex
        self.id = -1
        self.pose = np.copy(pose)
        self.sigma = np.copy(sigma)
        # dimension of the pose vector  # vector dimension.
        self.dim = pose.shape[0]

    def update_pose(self, pose):
        self.pose = pose




