from models.obstacles.OctagonObstacle import OctagonObstacle

class FeaturePoint(OctagonObstacle):

  def __init__(self, radius, pose, id):
    super(FeaturePoint, self).__init__(radius, pose)
    self.id = id