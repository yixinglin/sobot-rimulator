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


from math import *

from gui.ColorPalette import *


class Painter:

    def __init__(self, pixels_per_meter):
        """
        Initializes a Painter object
        :param pixels_per_meter: Specifies the amount of pixels contained in one meter
        """
        self.pixels_per_meter = pixels_per_meter

    def draw_frame(self, frame, widget, context):
        """
        Draws a frame onto a widget
        :param frame: The frame objects that will be drawn
        :param widget: The widget onto which the frame is drawn
        :param context: The cairo context to be used
        """
        width_pixels = widget.get_allocated_width()
        height_pixels = widget.get_allocated_height()

        # transform the the view to metric coordinates
        context.translate(width_pixels / 2.0, height_pixels / 2.0)  # move origin to center of window
        context.scale(self.pixels_per_meter,
                      -self.pixels_per_meter)  # pull view to edges of window ( also flips y-axis )

        # draw the background in white
        self.set_color(context, 'white', 1.0)
        context.paint()

        draw_list = frame.draw_list
        for component in draw_list:
            if component['type'] == 'ellipse':
                self.draw_ellipse(context,
                                  component['pos'],
                                  component['angle'],
                                  component['radius_x'],
                                  component['radius_y'],
                                  component['color'],
                                  component['alpha'])

            if component['type'] == 'circle':
                self.draw_circle(context,
                                 component['pos'],
                                 component['radius'],
                                 component['color'],
                                 component['alpha'])

            elif component['type'] == 'polygons':
                self.draw_polygons(context,
                                   component['polygons'],
                                   component['color'],
                                   component['alpha'])

            elif component['type'] == 'lines':
                self.draw_lines(context,
                                component['lines'],
                                component['linewidth'],
                                component['color'],
                                component['alpha'])
            elif component['type'] == 'rectangle':
                self.draw_rectangle(context,
                                    component['pos'],
                                    component['width'],
                                    component['height'],
                                    component['color'],
                                    component['alpha'])

    def draw_ellipse(self, context,
                     pos, angle,
                     radius_x, radius_y,
                     color, alpha):
        """
        Draws an ellipse
        :param context: The cairo context to be used
        :param pos: The position of the center of the ellipse
        :param angle: Angle of the x side
        :param radius_x: "Radius" of one ellipse side
        :param radius_y: "Radius" of the other ellipse side, perpendicular to the radius_x
        :param color: Color of the ellipse
        :param alpha: Alpha value of the color of the ellipse
        """
        if radius_x > 0 and radius_y > 0:
            self.set_color(context, color, alpha)
            context.translate(pos[0], pos[1])
            context.rotate(angle)
            context.scale(radius_x, radius_y)
            context.arc(0, 0, 1, 0, 2.0 * pi)
            context.fill()

    def draw_circle(self, context,
                    pos, radius,
                    color, alpha):
        """
        Draws a circle
        :param context: The cairo context to be used
        :param pos: The position of the center of the circle
        :param radius: The radius of the circle
        :param color: Color of the circle
        :param alpha: Alphave value of the color of the circle
        """
        self.set_color(context, color, alpha)
        context.arc(pos[0], pos[1], radius, 0, 2.0 * pi)
        context.fill()

    def draw_polygons(self, context,
                      polygons,
                      color, alpha):
        """
        Draws a filled out polygon
        :param context: The cairo context to be used
        :param polygons: List of polygon vertices
        :param color: Color of the polygon
        :param alpha: Alpha value of the color of the polygon
        """
        self.set_color(context, color, alpha)
        for polygon in polygons:
            context.new_path()
            context.move_to(*polygon[0])
            for point in polygon[1:]:
                context.line_to(*point)
            context.fill()

    def draw_lines(self, context,
                   lines, linewidth,
                   color, alpha):
        """
        Draws lines
        :param context: The cairo context to be used
        :param lines: List of line vertices
        :param linewidth: Width of the lines
        :param color: Color of the lines
        :param alpha: Alpha value of the color of the lines
        """
        self.set_color(context, color, alpha)
        context.set_line_width(linewidth)
        for line in lines:
            context.new_path()
            context.move_to(*line[0])
            for point in line[1:]:
                context.line_to(*point)
            context.stroke()

    def set_color(self, cairo_context, color_string, alpha):
        """
        Sets a color of the cairo context
        :param cairo_context: Cairo context to be modified
        :param color_string: String specifying the color
        :param alpha: Alpha value of the color
        """
        ColorPalette.dab(cairo_context, color_string, alpha)


    def draw_rectangle(self, context, pos, width, height, color, alpha):
        """
        Draws an rectangle
        :param context: The cairo context to be used
        :param pos: Top left coordinate of the rectangle
        :param width: Width of the rectangle
        :param height: Height of the rectangle
        :param color: Color of the rectangle, it's a tuple of rgb values (r, g, b) in range of 0.0 - 1.0
        :param alpha: Alpha value of the color of the rectangle
        """
        context.set_source_rgba(color[0], color[1], color[2], alpha)
        context.rectangle(pos[0], pos[1], width, height)
        context.fill()
