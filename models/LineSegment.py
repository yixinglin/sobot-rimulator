# Sobot Rimulator - A Robot Programming Tool
# Copyright (C) 2013-2014 Nicholas S. D. McCrea
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# Email mccrea.engineering@gmail.com for questions, comments, or to report bugs.


from models.Geometry import *
from utils import linalg2_util as linalg


class LineSegment(Geometry):

    def __init__(self, vertexes):
        """
        Initializes a LineSegment object
        :param vertexes: The vertices of the line segment
        """
        self.vertexes = vertexes  # the beginning and ending points of this line segment

        # define the centerpoint and radius of a circle containing this line segment
        # value is a tuple of the form ( [ cx, cy ], r )
        self.bounding_circle = self._bounding_circle()

    def get_transformation_to_pose(self, pose):
        """
        :param pose: The pose that this line segment should be transformed to
        :return: A copy of this line segment transformed to the given pose
        """
        p_pos, p_theta = pose.vunpack()
        return LineSegment(linalg.rotate_and_translate_vectors(self.vertexes, p_theta, p_pos))

    def _bounding_circle(self):
        """
        :return: The centerpoint and radius of a circle that fully contains this line segment
        """
        v = self._as_vector()
        vhalf = linalg.scale(v, 0.5)

        c = linalg.add(self.vertexes[0], vhalf)
        r = linalg.mag(v) * 0.5

        return c, r

    def _as_vector(self):
        """
        :return: The vector from the beginning point to the end point of this line segment
        """
        return linalg.sub(self.vertexes[1], self.vertexes[0])
