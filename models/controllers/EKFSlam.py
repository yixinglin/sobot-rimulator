import numpy as np
from math import pi


class EKFSlam:

    def __init__(self, supervisor_interface):
        # bind the supervisor
        self.supervisor = supervisor_interface

        # sensor placements
        self.proximity_sensor_placements = self.supervisor.proximity_sensor_placements()

        # sensor gains (weights)
        self.sensor_gains = [1.0 + ((0.4 * abs(p.theta)) / pi)
                             for p in self.supervisor.proximity_sensor_placements()]

        # key vectors and data (initialize to any non-zero vector)
        self.obstacle_vectors = [[1.0, 0.0]] * len(self.proximity_sensor_placements)

        self.landmarks = []
