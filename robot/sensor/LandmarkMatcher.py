"""
This sensor provides a set of correspondences of landmarks.
To apply this class, we assume that the data associations of landmarks are known.
A "landmark matcher" can be a camera that can extract features from the environment,
also scan-matching algorithms are apply to match those features.
"""
from robot.sensor.Sensor import *

class LandmarkMatcher(Sensor):

  def __init__(self):
    self.matcher = dict()

  def detect(self, sensor_id, landmark_id):
    self.matcher[sensor_id] = landmark_id

  def read(self):
    return self.matcher


