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


import gi

gi.require_version('Gtk', '3.0')
from gi.repository import Gtk as gtk

from gui.Frame import Frame
from gui.Painter import Painter
from matplotlib import pyplot as plt
# user response codes for file chooser dialog buttons
LS_DIALOG_RESPONSE_CANCEL = 1
LS_DIALOG_RESPONSE_ACCEPT = 2


class Viewer:

    def __init__(self, simulator, viewer_config, num_frames, slam_cfg):
        """
        Initializes a Viewer object
        :param simulator: The underlying simulator
        :param viewer_config: The configuration of the Viewer
        :param num_frames: Number of frame of the GUI, determined by which algorithms are activated
        :param ekf_enabled: Boolean value specifying if EKF is enabled
        :param use_slam_evaluation: Boolean value specifying if the slam evaluation is enabled
        """
        # bind the simulator
        self.simulator = simulator

        self.cfg = viewer_config
        self.ekf_enabled = slam_cfg["ekf_slam"]["enabled"]
        self.fastslam_enabled = slam_cfg["fast_slam"]["enabled"]
        self.graphslambased_enabled = slam_cfg["graph_based_slam"]["enabled"]
        self.use_slam_evaluation = slam_cfg["evaluation"]["enabled"]
        self.mapping_enabled = slam_cfg["mapping"]["enabled"]

        # initialize camera parameters
        self.num_frames = num_frames
        self.view_width_pixels = viewer_config["pixels_width"]
        self.view_height_pixels = viewer_config["pixels_height"]
        self.pixels_per_meter = viewer_config["zoom"]

        # initialize frames
        self.current_frames = [Frame() for _ in range(self.num_frames)]

        # initialize the window
        self.window = gtk.Window()
        self.window.set_title('Sobot Rimulator')
        self.window.set_resizable(False)
        self.window.connect('delete_event', self.on_delete)

        label_strings = ["World"]
        if self.ekf_enabled:
            label_strings += ['EKF SLAM']
        if self.fastslam_enabled:
            label_strings += ['FAST SLAM']
        if self.graphslambased_enabled:
            label_strings += ['Graph-based SLAM']

        label_strings = label_strings[:self.num_frames]

        self.labels = []
        for label_string in label_strings:
            label = gtk.Label()
            label.set_text(label_string)
            self.labels.append(label)

        # initialize the drawing_areas
        self.drawing_areas = []
        # This list contains the drawing functions for the frames. The list has same length as number of frames.
        on_expose_functions = [self.on_expose1, self.on_expose2, self.on_expose3, self.on_expose4][:self.num_frames]
        for on_expose in on_expose_functions:
            drawing_area = gtk.DrawingArea()
            drawing_area.set_size_request(self.view_width_pixels, self.view_height_pixels)
            drawing_area.connect('draw', on_expose)
            self.drawing_areas.append(drawing_area)

        # initialize the painter
        self.painter = Painter(self.pixels_per_meter)

    # == initialize the buttons

        # build the play button
        self.button_play = gtk.Button('Play')
        play_image = gtk.Image()
        play_image.set_from_stock(gtk.STOCK_MEDIA_PLAY, gtk.IconSize.BUTTON)
        self.button_play.set_image(play_image)
        self.button_play.set_image_position(gtk.PositionType.LEFT)
        self.button_play.connect('clicked', self.on_play)

        # build the stop button
        self.button_stop = gtk.Button('Stop')
        stop_image = gtk.Image()
        stop_image.set_from_stock(gtk.STOCK_MEDIA_STOP, gtk.IconSize.BUTTON)
        self.button_stop.set_image(stop_image)
        self.button_stop.set_image_position(gtk.PositionType.LEFT)
        self.button_stop.connect('clicked', self.on_stop)

        # build the step button
        self.button_step = gtk.Button('Step')
        step_image = gtk.Image()
        step_image.set_from_stock(gtk.STOCK_MEDIA_NEXT, gtk.IconSize.BUTTON)
        self.button_step.set_image(step_image)
        self.button_step.set_image_position(gtk.PositionType.LEFT)
        self.button_step.connect('clicked', self.on_step)

        # build the reset button
        self.button_reset = gtk.Button('Reset')
        reset_image = gtk.Image()
        reset_image.set_from_stock(gtk.STOCK_MEDIA_REWIND, gtk.IconSize.BUTTON)
        self.button_reset.set_image(reset_image)
        self.button_reset.set_image_position(gtk.PositionType.LEFT)
        self.button_reset.connect('clicked', self.on_reset)

        # build the save map button
        self.button_save_map = gtk.Button('Save Map')
        save_map_image = gtk.Image()
        save_map_image.set_from_stock(gtk.STOCK_SAVE, gtk.IconSize.BUTTON)
        self.button_save_map.set_image(save_map_image)
        self.button_save_map.set_image_position(gtk.PositionType.LEFT)
        self.button_save_map.connect('clicked', self.on_save_map)

        # build the load map button
        self.button_load_map = gtk.Button('Load Map')
        load_map_image = gtk.Image()
        load_map_image.set_from_stock(gtk.STOCK_OPEN, gtk.IconSize.BUTTON)
        self.button_load_map.set_image(load_map_image)
        self.button_load_map.set_image_position(gtk.PositionType.LEFT)
        self.button_load_map.connect('clicked', self.on_load_map)

        # build the random map buttons
        self.button_random_map = gtk.Button('Random Map')
        random_map_image = gtk.Image()
        random_map_image.set_from_stock(gtk.STOCK_REFRESH, gtk.IconSize.BUTTON)
        self.button_random_map.set_image(random_map_image)
        self.button_random_map.set_image_position(gtk.PositionType.LEFT)
        self.button_random_map.connect('clicked', self.on_random_map)

        # build the draw-invisibles toggle button
        self.draw_invisibles = False  # controls whether invisible world elements are displayed
        self.button_draw_invisibles = gtk.Button()
        self._decorate_draw_invisibles_button_inactive()
        self.button_draw_invisibles.set_image_position(gtk.PositionType.LEFT)
        self.button_draw_invisibles.connect('clicked', self.on_draw_invisibles)

        # build the plot slam evaluation button
        self.button_slam_evaluation = gtk.Button("Plot Slam Evaluation")
        self.button_slam_evaluation.set_image_position(gtk.PositionType.LEFT)
        self.button_slam_evaluation.connect('clicked', self.on_slam_evaluation)

        # build the plot-covariance-matrix button
        self.button_plot_covariances = gtk.Button("Plot Covariance Matrix")
        self.button_plot_covariances.set_image_position(gtk.PositionType.LEFT)
        self.button_plot_covariances.connect('clicked', self.on_plot_covariances)

        # build the plot-graph button
        self.button_plot_graph = gtk.Button("Plot Graph")
        self.button_plot_graph.set_image_position(gtk.PositionType.LEFT)
        self.button_plot_graph.connect('clicked', self.on_plot_graph)

        # build the start-mapping button
        self.start_mapping = False  # controls whether mapping is executed.
        self.button_start_mapping = gtk.Button("Start Mapping")
        self._decorate_start_mapping_button_inactive()
        self.button_start_mapping.set_image_position(gtk.PositionType.LEFT)
        self.button_start_mapping.connect('clicked', self.on_start_mapping)

        # == lay out the window

        labels_box = gtk.HBox(spacing=self.view_width_pixels - 52)  # Subtract number of pixels that the text of the labels roughly need
        for label in self.labels:
            labels_box.pack_start(label, False, False, 0)
        labels_alignment = gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        labels_alignment.add(labels_box)

        plots_box = gtk.HBox(spacing=5)
        for drawing_area in self.drawing_areas:
            plots_box.pack_start(drawing_area, False, False, 0)
        plots_alignment = gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        plots_alignment.add(plots_box)

        # pack the simulation control buttons
        sim_controls_box = gtk.HBox(spacing=5)
        sim_controls_box.pack_start(self.button_play, False, False, 0)
        sim_controls_box.pack_start(self.button_stop, False, False, 0)
        sim_controls_box.pack_start(self.button_step, False, False, 0)
        sim_controls_box.pack_start(self.button_reset, False, False, 0)

        # pack the map control buttons
        map_controls_box = gtk.HBox(spacing=5)
        map_controls_box.pack_start(self.button_save_map, False, False, 0)
        map_controls_box.pack_start(self.button_load_map, False, False, 0)
        map_controls_box.pack_start(self.button_random_map, False, False, 0)

        # pack the information buttons
        information_box = gtk.HBox()
        information_box.pack_start(self.button_draw_invisibles, False, False, 0)
        if self.ekf_enabled:
            information_box.pack_start(self.button_plot_covariances, False, False, 0)

        if self.graphslambased_enabled:
            information_box.pack_start(self.button_plot_graph, False, False, 0)

        if self.mapping_enabled:
            information_box.pack_start(self.button_start_mapping, False, False, 0)

        if num_frames > 1 and self.use_slam_evaluation:
            information_box.pack_start(self.button_slam_evaluation, False, False, 0)

        # align the controls
        sim_controls_alignment = gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        map_controls_alignment = gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        invisibles_button_alignment = gtk.Alignment(xalign=0.5, yalign=0.5, xscale=0, yscale=0)
        sim_controls_alignment.add(sim_controls_box)
        map_controls_alignment.add(map_controls_box)
        invisibles_button_alignment.add(information_box)

        # create the alert box
        self.alert_box = gtk.Label()

        # lay out the simulation view and all of the controls
        layout_box = gtk.VBox()
        layout_box.pack_start(labels_alignment, False, False, 5)
        layout_box.pack_start(plots_alignment, False, False, 0)
        layout_box.pack_start(self.alert_box, False, False, 5)
        layout_box.pack_start(sim_controls_alignment, False, False, 5)
        layout_box.pack_start(map_controls_alignment, False, False, 5)
        layout_box.pack_start(invisibles_button_alignment, False, False, 5)

        # apply the layout
        self.window.add(layout_box)

        # show the simulator window
        self.window.show_all()

    def new_frame(self):
        """
        Initialiues empty frames
        """
        self.current_frames = [Frame() for _ in range(self.num_frames)]

    def draw_frame(self):
        """
        Draws the frames
        """
        for drawing_area in self.drawing_areas:
            drawing_area.queue_draw_area(0, 0, self.view_width_pixels, self.view_height_pixels)

    def control_panel_state_init(self):
        """
        Specifies the button sensitivities at the initial state
        """
        self.alert_box.set_text('')
        self.button_play.set_sensitive(True)
        self.button_stop.set_sensitive(False)
        self.button_step.set_sensitive(True)
        self.button_reset.set_sensitive(False)

    def control_panel_state_playing(self):
        """
        Specifies the button sensitivities while the simulation is running
        """
        self.button_play.set_sensitive(False)
        self.button_stop.set_sensitive(True)
        self.button_reset.set_sensitive(True)

    def control_panel_state_paused(self):
        """
        Specifies the button sensitivities while the simulation is paused
        """
        self.button_play.set_sensitive(True)
        self.button_stop.set_sensitive(False)
        self.button_reset.set_sensitive(True)

    def control_panel_state_finished(self, alert_text):
        """
        Specifies the button sensitivies once the simulation encountered an exception
        :param alert_text: Text to be displayed to the user
        """
        self.alert_box.set_text(alert_text)
        self.button_play.set_sensitive(False)
        self.button_stop.set_sensitive(False)
        self.button_step.set_sensitive(False)

    # EVENT HANDLERS:
    def on_play(self, widget):
        """
        Callback function that handles a click on the "Play" button
        :param widget: The corresponding widget
        """
        self.simulator.play_sim()

    def on_stop(self, widget):
        """
        Callback function that handles a click on the "Stop" button
        :param widget: The corresponding widget
        """
        self.simulator.pause_sim()

    def on_step(self, widget):
        """
        Callback function that handles a click on the "Step" button
        :param widget: The corresponding widget
        """
        self.simulator.step_sim_once()

    def on_reset(self, widget):
        """
        Callback function that handles a click on the "Reset" button
        :param widget: The corresponding widget
        """
        self.simulator.reset_sim()

    def on_save_map(self, widget):
        """
        Callback function that handles a click on the "Save Map" button
        :param widget: The corresponding widget
        """
        # create the file chooser
        file_chooser = gtk.FileChooserDialog(title='Save Map',
                                             parent=self.window,
                                             action=gtk.FileChooserAction.SAVE,
                                             buttons=(gtk.STOCK_CANCEL, LS_DIALOG_RESPONSE_CANCEL,
                                                      gtk.STOCK_SAVE, LS_DIALOG_RESPONSE_ACCEPT))
        file_chooser.set_do_overwrite_confirmation(True)
        file_chooser.set_current_folder('maps')

        # run the file chooser dialog
        response_id = file_chooser.run()

        # handle the user's response
        if response_id == LS_DIALOG_RESPONSE_CANCEL:
            file_chooser.destroy()
        elif response_id == LS_DIALOG_RESPONSE_ACCEPT:
            self.simulator.save_map(file_chooser.get_filename())
            file_chooser.destroy()

    def on_load_map(self, widget):
        """
        Callback function that handles a click on the "Load map" button
        :param widget: The corresponding widget
        """
        # create the file chooser
        file_chooser = gtk.FileChooserDialog(title='Load Map',
                                             parent=self.window,
                                             action=gtk.FileChooserAction.SAVE,
                                             buttons=(gtk.STOCK_CANCEL, LS_DIALOG_RESPONSE_CANCEL,
                                                      gtk.STOCK_OPEN, LS_DIALOG_RESPONSE_ACCEPT))
        file_chooser.set_current_folder('maps')

        # run the file chooser dialog
        response_id = file_chooser.run()

        # handle the user's response
        if response_id == LS_DIALOG_RESPONSE_CANCEL:
            file_chooser.destroy()
        elif response_id == LS_DIALOG_RESPONSE_ACCEPT:
            self.simulator.load_map(file_chooser.get_filename())
            file_chooser.destroy()

    def on_random_map(self, widget):
        """
        Callback function that handles a click on the "Random map" button
        :param widget: The corresponding widget
        """
        self.simulator.random_map()

    def on_draw_invisibles(self, widget):
        """
        Callback function that handles a click on the "Draw invisibles" button
        :param widget: The corresponding widget
        """
        # toggle the draw_invisibles state
        self.draw_invisibles = not self.draw_invisibles
        if self.draw_invisibles:
            self._decorate_draw_invisibles_button_active()
        else:
            self._decorate_draw_invisibles_button_inactive()
        self.simulator.draw_world()

    def on_slam_evaluation(self, widget):
        """
        Callback function that handles a click on the "Slam evaluation" button
        :param widget: The corresponding widget
        """

        if self.simulator.slam_evaluations is not None:
            self.simulator.slam_evaluations.plot()


    def on_plot_covariances(self, widget):
        """
        Callback function that handles a click on the "Plot covariances" button
        :param widget: The corresponding widget
        """
        self.simulator.ekfslam_plotter.plot_covariances()

    def on_plot_graph(self, widget):
        """
        Callback function that handles a click on the "Plot Graph" button
        :param widget: The corresponding widget
        """
        self.simulator.graphbasedslam_plotter.plot_graph()

    def on_start_mapping(self, widget):
        """
        Callback function that handles a click on the "Start Mapping" button
        :param widget: The corresponding widget
        :return:
        """
        self.start_mapping = not self.start_mapping
        if self.start_mapping:
            self._decorate_start_mapping_button_active()
            self.simulator.draw_world()
        else:
            self._decorate_start_mapping_button_inactive()
            self.simulator.reset_grid_map()


    def on_expose1(self, widget, context):
        """
        Draws the first frame
        :param widget: The corresponding widget
        :param context: The cairo context to be used
        """
        self.painter.draw_frame(self.current_frames[0], widget, context)

    def on_expose2(self, widget, context):
        """
        Draws the second frame
        :param widget: The corresponding widget
        :param context: The cairo context to be used
        """
        self.painter.draw_frame(self.current_frames[1], widget, context)

    def on_expose3(self, widget, context):
        """
        Draws the third frame
        :param widget: The corresponding widget
        :param context: The cairo context to be used
        """
        self.painter.draw_frame(self.current_frames[2], widget, context)

    def on_expose4(self, widget, context):
        """
        Draws the fourth frame
        :param widget: The corresponding widget
        :param context: The cairo context to be used
        """
        self.painter.draw_frame(self.current_frames[3], widget, context)

    def on_delete(self, widget, event):
        """
        Callback function the handles a delete event
        :param widget: The corresponding widget
        :param event: An event to be handled
        :return:
        """
        gtk.main_quit()
        return False

    def _decorate_draw_invisibles_button_active(self):
        """
        Specifies the "Draw invisibles" button while it is enabled
        """
        draw_invisibles_image = gtk.Image()
        draw_invisibles_image.set_from_stock(gtk.STOCK_REMOVE, gtk.IconSize.BUTTON)
        self.button_draw_invisibles.set_image(draw_invisibles_image)
        self.button_draw_invisibles.set_label('Hide Invisibles')

    def _decorate_draw_invisibles_button_inactive(self):
        """
        Specifies the "Draw invisibles" button while it is disabled
        """
        draw_invisibles_image = gtk.Image()
        draw_invisibles_image.set_from_stock(gtk.STOCK_ADD, gtk.IconSize.BUTTON)
        self.button_draw_invisibles.set_image(draw_invisibles_image)
        self.button_draw_invisibles.set_label('Show Invisibles')

    def _decorate_start_mapping_button_inactive(self):
        """
        Specifies the "Start Mapping" button while it is disabled
        """
        draw_invisibles_image = gtk.Image()
        draw_invisibles_image.set_from_stock(gtk.STOCK_ADD, gtk.IconSize.BUTTON)
        self.button_start_mapping.set_image(draw_invisibles_image)
        self.button_start_mapping.set_label('Start Mapping')

    def _decorate_start_mapping_button_active(self):
        """
        Specifies the "Start Mapping" button while it is enabled
        """
        draw_invisibles_image = gtk.Image()
        draw_invisibles_image.set_from_stock(gtk.STOCK_ADD, gtk.IconSize.BUTTON)
        self.button_start_mapping.set_image(draw_invisibles_image)
        self.button_start_mapping.set_label('Reset Mapping')