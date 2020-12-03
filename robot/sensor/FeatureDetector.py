"""
This sensor provides a set of identifiers of landmarks.
To apply this class, we assume that the data associations of landmarks are known.
A "Feature Detector" can be a camera that can extract features from the environment.
"""
from robot.sensor.Sensor import *

class FeatureDetector(Sensor):

  def __init__(self):
    """
    Identifiers: a dictionary that maps sensor identifiers to feature identifiers
    """
    self.identifiers = dict()

  def detect(self, sensor_id, feature_id):
    self.identifiers[sensor_id] = feature_id

  def read(self):
    return self.identifiers


