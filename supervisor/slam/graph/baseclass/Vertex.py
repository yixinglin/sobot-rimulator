import numpy as np

class Vertex:

    def __init__(self, pose, sigma, observation = None):

        self.id = -1
        self.pose = np.copy(pose)
        self.observation = np.copy(observation)
        self.sigma = np.copy(sigma)
        self.dim = pose.shape[0]  # dimension of the pose vector

    def update_pose(self, pose):
        self.pose = pose




