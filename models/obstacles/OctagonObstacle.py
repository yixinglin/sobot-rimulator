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


from models.Polygon import *


class OctagonObstacle:

    def __init__(self, radius, pose):
        self.pose = pose
        self.radius = radius

        # define the geometry

        angled_length = radius * (1 / 2 ** 0.5)
        vertexes = [[0, radius],
                    [angled_length, angled_length],
                    [radius, 0],
                    [angled_length, -angled_length],
                    [0, -radius],
                    [-angled_length, -angled_length],
                    [-radius, 0],
                    [-angled_length, angled_length]]
        self.geometry = Polygon(vertexes)
        self.global_geometry = Polygon(vertexes).get_transformation_to_pose(self.pose)