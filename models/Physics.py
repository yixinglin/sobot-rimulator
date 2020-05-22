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


import utils.geometrics_util as geometrics

from simulation.exceptions import CollisionException


class Physics:

    def __init__(self, world):
        """
        Initializes a Physics object
        :param world: The world this physics engine acts on
        """
        self.world = world

    def apply_physics(self):
        """
        Apply the physics interactions of the world.
        Consists of collision detections and updating the robots proximity sensors
        """
        self._detect_collisions()
        self._update_proximity_sensors()

    def _detect_collisions(self):
        """
        Checks the world for any collisions between colliding objects (robots) and solid objects (obstacles)
        Raises a CollisionException if a collision is detected
        """
        colliders = self.world.colliders()
        solids = self.world.solids()

        for collider in colliders:
            polygon1 = collider.global_geometry  # polygon1

            for solid in solids:
                if solid is not collider:  # don't test an object against itself
                    polygon2 = solid.global_geometry  # polygon2

                    if geometrics.check_nearness(polygon1,
                                                 polygon2):  # don't bother testing objects that are not near each other
                        if geometrics.convex_polygon_intersect_test(polygon1, polygon2):
                            raise CollisionException()

    def _update_proximity_sensors(self):
        """
        Update any proximity sensors that are in range of solid objects
        """
        robots = self.world.robots
        solids = self.world.solids()

        for robot in robots:
            sensors = robot.ir_sensors

            for sensor in sensors:
                dmin = float('inf')
                detector_line = sensor.detector_line

                for solid in solids:

                    if solid is not robot:  # assume that the sensor does not detect it's own robot
                        solid_polygon = solid.global_geometry

                        if geometrics.check_nearness(detector_line,
                                                     solid_polygon):  # don't bother testing objects that are not near each other
                            intersection_exists, intersection, d = geometrics.directed_line_segment_polygon_intersection(
                                detector_line, solid_polygon)

                            if intersection_exists and d < dmin:
                                dmin = d

                # if there is an intersection, update the sensor with the new delta value
                if dmin != float('inf'):
                    sensor.detect(dmin)
                else:
                    sensor.detect(None)
