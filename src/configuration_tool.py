import tkinter as tk
from tkinter import ttk, filedialog

from dtrack_params import dtrack_params
from project import project_file

class ConfigurationTool(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Configure window
        self.title("DTrack2 options")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.__frm_controls = tk.Frame(self)
        self.__frm_controls.rowconfigure(0, weight=1)
        self.__btn_confirm = tk.Button(self.__frm_controls,
                                       text='Confirm',
                                       command=self.__confirm_callback)
        self.__btn_cancel = tk.Button(self.__frm_controls,
                                      text='Cancel',
                                      command=self.destroy)
        
        self.__frm_dtrack_options = tk.Frame(self)

        self.__frm_dtrack_options.grid(row=0, column=0, sticky='nesw')
        self.__frm_controls.grid(row=1,column=0, sticky='nesw')

        self.__btn_confirm.grid(column=0, row=0, sticky='nws')
        self.__btn_cancel.grid(column=1, row=0, sticky='nws')
        
        self.__arrange_dtrack_options()

        # Set minsize to whatever the final window size is with all elements in place
        self.update()
        self.minsize(self.winfo_width(),self.winfo_height())
        self.maxsize(self.winfo_width(),self.winfo_height())        

    def __config_rows_and_cols(self, 
                               widget, 
                               n_rows, 
                               n_cols, 
                               weight=1,
                               exceptions=[]):
        """
        Configure weight for all rows and columns in a widget (expected to be
        frame or labelframe).

        :param widget: The widget
        :param n_rows: The number of rows
        :param n_cols: The number of columns
        :param weight: The weight to set
        :param exceptions: A list of tuples of the form 
        """
        for r in range(n_rows):
            for c in range(n_cols):
                widget.rowconfigure(r, weight=weight)
                widget.columnconfigure(c, weight=weight)

    def __arrange_dtrack_options(self):
        """
        Define all widgets for the DTrack2 options panel and 'grid' them for display.
        """
        # Project options pane, one row for each section, one column.
        self.__config_rows_and_cols(self.__frm_dtrack_options,
                                    n_rows=2,
                                    n_cols=1,
                                    weight=1)

        # Project options info-string
        outstr = "This pane allows you to configure options for DTrack2."
        lbl_project_option_info = tk.Label(self.__frm_dtrack_options,
                                           text=outstr, 
                                           anchor='w',
                                           relief='sunken')
        lbl_project_option_info.grid(column=0, row=0, sticky='new', pady=10, padx=10)      


        # Construct and arrange option frames
        lbf_tracker_options = tk.LabelFrame(self.__frm_dtrack_options,
                                            text="Autotracker", padx=10)
        lbf_autocalibration_options = tk.LabelFrame(self.__frm_dtrack_options,
                                                    text="Autocalibration", padx=10)
        lbf_video_selection_options = tk.LabelFrame(self.__frm_dtrack_options,
                                                    text="Video selection")        
        lbf_processing_options = tk.LabelFrame(self.__frm_dtrack_options,
                                               text="Track processing", padx=10)        
        
        lbf_video_selection_options.grid(column=0, row=1, sticky='nesw')
        lbf_autocalibration_options.grid(column=0, row=3, sticky='nesw')
        lbf_tracker_options.grid(column=0, row=4, sticky='nesw')
        lbf_processing_options.grid(column=0, row=5, sticky='nesw')

        #
        # Video selection options
        #
        self.__config_rows_and_cols(lbf_video_selection_options,
                                    n_rows=1,
                                    n_cols=3)
        self.__stv_video_directory = tk.StringVar()
        self.__stv_video_directory.set(dtrack_params["options.video.directory"])
        lbl_video_directory = tk.Label(lbf_video_selection_options,
                                       text="Default video directory: ",
                                       anchor='w')
        ent_video_directory = tk.Entry(lbf_video_selection_options,
                                       width=80,
                                       textvariable=self.__stv_video_directory)
        btn_select_video_directory = tk.Button(lbf_video_selection_options,
                                               text='Select',
                                               command=self.__select_video_directory_callback)
                                       
        lbl_video_directory.grid(column=0, row=0, sticky='ew')
        ent_video_directory.grid(column=1, row=0, sticky='ew')
        btn_select_video_directory.grid(column=2, row=0, sticky='ew')

        #
        # Autocalibration options
        #
        lbf_autocalibration_options.columnconfigure(0, weight=1)
        self.__blv_fix_k1 = tk.BooleanVar()
        self.__blv_fix_k2 = tk.BooleanVar()
        self.__blv_fix_k3 = tk.BooleanVar()
        self.__blv_fix_tangential = tk.BooleanVar()
        self.__blv_show_meta_text = tk.BooleanVar()

        self.__blv_fix_k1.set(dtrack_params["options.autocalibration.fix_k1"])
        self.__blv_fix_k2.set(dtrack_params["options.autocalibration.fix_k2"])
        self.__blv_fix_k3.set(dtrack_params["options.autocalibration.fix_k3"])
        self.__blv_show_meta_text.set(dtrack_params["options.autocalibration.show_meta_text"])
        self.__blv_fix_tangential.set(dtrack_params["options.autocalibration.fix_tangential"])

        chb_show_metainformation = tk.Checkbutton(lbf_autocalibration_options,
                                                  text="Show default metainformation text",
                                                  variable=self.__blv_show_meta_text)
        
        lbl_autocalibration_info = tk.Label(lbf_autocalibration_options,
                                            text="Only change the options below if you know what they do!",
                                            anchor='w',
                                            fg='#eb3a34',
                                            padx=10,
                                            pady=10,
                                            relief='sunken')
        chb_fix_k1 = tk.Checkbutton(lbf_autocalibration_options,
                                    text="Fix K1",
                                    variable=self.__blv_fix_k1)
        chb_fix_k2 = tk.Checkbutton(lbf_autocalibration_options,
                                    text="Fix K2",
                                    variable=self.__blv_fix_k2)
        chb_fix_k3 = tk.Checkbutton(lbf_autocalibration_options,
                                    text="Fix K3",
                                    variable=self.__blv_fix_k3)
        chb_fix_tangential = tk.Checkbutton(lbf_autocalibration_options,
                                            text="Fix tangential",
                                            variable=self.__blv_fix_tangential)
        
        chb_show_metainformation.grid(row=0, column=0, sticky='nw')
        lbl_autocalibration_info.grid(row=1, column=0, sticky='nesw')
        chb_fix_k1.grid(row=2, column=0, sticky='nw')
        chb_fix_k2.grid(row=3, column=0, sticky='nw')
        chb_fix_k3.grid(row=4, column=0, sticky='nw')
        chb_fix_tangential.grid(row=5, column=0, sticky='nw')


        #
        # Autotracker options
        #        
        self.__stv_dtrack_track_point = tk.StringVar()
        self.__stv_cv_backend = tk.StringVar()
        self.__stv_bg_computation_method = tk.StringVar()
        self.__stv_bg_sample_size = tk.StringVar()
        self.__stv_track_interval = tk.StringVar()
        self.__blv_remember_roi = tk.BooleanVar()
        self.__blv_show_roi = tk.BooleanVar()

        self.__stv_dtrack_track_point.set(dtrack_params["options.autotracker.track_point"])
        self.__stv_cv_backend.set(dtrack_params["options.autotracker.cv_backend"])
        self.__stv_bg_computation_method.set(dtrack_params["options.autotracker.bg_computation_method"])
        self.__stv_bg_sample_size.set(str(dtrack_params["options.autotracker.bg_sample_size"]))
        self.__stv_track_interval.set(str(dtrack_params["options.autotracker.track_interval"]))
        self.__blv_remember_roi.set(dtrack_params["options.autotracker.remember_roi"])
        self.__blv_show_roi.set(dtrack_params["options.autotracker.show_roi"])

        lbl_track_point_selection = tk.Label(lbf_tracker_options,
                                             text="Default autotracker target: ",
                                             anchor='w')
        cmb_track_point_selection = ttk.Combobox(lbf_tracker_options,
                                                 values=["centre-of-mass",
                                                         "centre-of-bounding-box"],
                                                 state="readonly",
                                                 textvariable=self.__stv_dtrack_track_point)
        cmb_track_point_selection.set(self.__stv_dtrack_track_point.get())

        lbl_cv_backend = tk.Label(lbf_tracker_options,
                                             text="OpenCV tracking backend: ",
                                             anchor='w')
        cmb_cv_backend= ttk.Combobox(lbf_tracker_options,
                                               values=["BOOSTING",
                                                       "MIL",
                                                       "KCF",
                                                       "TLD",
                                                       "MEDIANFLOW",
                                                       "GOTURN",
                                                       "MOSSE",
                                                       "CSRT"],
                                               state="readonly",
                                               textvariable=self.__stv_cv_backend)
        cmb_cv_backend.set(self.__stv_cv_backend.get())

        lbl_bg_computation_method = tk.Label(lbf_tracker_options,
                                             text="Background computation method: ",
                                             anchor='w')
        cmb_bg_computation_method = ttk.Combobox(lbf_tracker_options,
                                               values=["first_N_median",
                                                       "first_N_mean",
                                                       "random_N_mean"],
                                               state="readonly",
                                               textvariable=self.__stv_bg_computation_method)
        cmb_bg_computation_method.set(self.__stv_bg_computation_method.get())
    
        lbl_bg_sample_size = tk.Label(lbf_tracker_options,
                                      text="N frames to use for background: ",
                                      anchor='w')
        ent_bg_sample_size =  tk.Entry(lbf_tracker_options,
                                       textvariable=self.__stv_bg_sample_size,
                                       validate='key',
                                       validatecommand=(
                                           self.register(
                                               lambda input: input.isdigit() or input==''
                                               ), '%P'))
        ent_bg_sample_size.insert(0, self.__stv_bg_sample_size)

        lbl_track_interval = tk.Label(lbf_tracker_options,
                                      text="Tracking inverval: ",
                                      anchor="w")
        
        spb_track_interval = ttk.Spinbox(lbf_tracker_options,
                                         state='readonly',
                                         from_=1,
                                         to=20,
                                         textvariable=self.__stv_track_interval)
        
        chb_remember_roi = tk.Checkbutton(lbf_tracker_options,
                                          text='Remember ROI',
                                          variable=self.__blv_remember_roi
                                          )
        
        chb_show_roi = tk.Checkbutton(lbf_tracker_options,
                                      text="Show ROI window",
                                      variable=self.__blv_show_roi)

        lbl_track_point_selection.grid(column=0, row=0, sticky='nw')
        cmb_track_point_selection.grid(column=1, row=0, sticky='nw')

        lbl_cv_backend.grid(column=0, row=1, sticky='nw')
        cmb_cv_backend.grid(column=1, row=1, sticky='nw')

        lbl_bg_computation_method.grid(column=0, row=2, sticky='nw')
        cmb_bg_computation_method.grid(column=1, row=2, sticky='nw')

        lbl_bg_sample_size.grid(column=0, row=3, sticky='nw')
        ent_bg_sample_size.grid(column=1, row=3, sticky='nw')

        lbl_track_interval.grid(column=0, row=4, sticky='nw')
        spb_track_interval.grid(column=1, row=4, sticky='nw')

        chb_remember_roi.grid(column=0, row=5, sticky='nw')

        chb_show_roi.grid(column=0, row=6, sticky='nw')

        
        #
        # Track processing options
        #
        self.__blv_plot_grid = tk.BooleanVar()
        self.__blv_include_legend = tk.BooleanVar()
        self.__stv_plot_filename = tk.StringVar()
        self.__stv_plot_filetype = tk.StringVar()
        self.__blv_plot_zero_tracks = tk.BooleanVar()
        self.__stv_smoothing_spline_degree = tk.StringVar()
        self.__stv_smoothing_scale_factor = tk.StringVar()

        self.__blv_plot_grid.set(dtrack_params["options.processing.plot_grid"])
        self.__blv_include_legend.set(dtrack_params["options.processing.include_legend"])
        self.__stv_plot_filename.set(dtrack_params["options.processing.filename"])
        self.__stv_plot_filetype.set(dtrack_params["options.processing.filetype"])
        self.__blv_plot_zero_tracks.set(dtrack_params["options.processing.zero"])
        self.__stv_smoothing_spline_degree.set(str(dtrack_params["options.processing.smoothing_spline_degree"]))
        self.__stv_smoothing_scale_factor.set(str(dtrack_params["options.processing.smoothing_scale_factor"]))
        

        chb_plot_grid = tk.Checkbutton(lbf_processing_options,
                                       text="Plot grid",
                                       variable=self.__blv_plot_grid)
        chb_include_legend = tk.Checkbutton(lbf_processing_options,
                                            text="Include legend",
                                            variable=self.__blv_include_legend)
        
        lbl_plot_filename = tk.Label(lbf_processing_options,
                                     text="Plot filename: ",
                                     anchor='w')
        ent_plot_filename = tk.Entry(lbf_processing_options,
                                     textvariable=self.__stv_plot_filename)
        
        lbl_plot_filetype = tk.Label(lbf_processing_options,
                                     text="Plot file format: ",
                                     anchor='w')
        
        cmb_plot_filetype = ttk.Combobox(lbf_processing_options,
                                         values=['pdf',
                                                 'eps',
                                                 'svg',
                                                 'png (400dpi)'],
                                         textvariable=self.__stv_plot_filetype)
        
        cmb_plot_zero_tracks = tk.Checkbutton(lbf_processing_options,
                                              variable=self.__blv_plot_zero_tracks,
                                              text="All tracks start at origin")
        
        lbl_smoothing_spline_degree = tk.Label(lbf_processing_options,
                                               text="Smoothing spline degree: ")
        spb_smoothing_spline_degree = ttk.Spinbox(lbf_processing_options,
                                                  textvariable=self.__stv_smoothing_spline_degree,
                                                  from_=1,
                                                  to=5,
                                                  state='readonly')
        
        lbl_smoothing_scale_factor = tk.Label(lbf_processing_options,
                                              text="Smoothing scale factor: "
                                              )
        spb_smoothing_scale_factor = ttk.Spinbox(lbf_processing_options,
                                                 state='readonly',
                                                 from_=0,
                                                 to=1,
                                                 increment=0.01,
                                                 textvariable=self.__stv_smoothing_scale_factor)
        

        lbl_plot_filename.grid(row=0, column=0, sticky='nw')
        ent_plot_filename.grid(row=0, column=1, sticky='nw')
        lbl_plot_filetype.grid(row=1, column=0, sticky='nw')
        cmb_plot_filetype.grid(row=1, column=1, sticky='nw')
        chb_plot_grid.grid(row=2, column=0, sticky='nw')
        chb_include_legend.grid(row=3, column=0, sticky='nw')
        cmb_plot_zero_tracks.grid(row=4, column=0, columnspan=2, sticky='nw')
        lbl_smoothing_spline_degree.grid(row=5, column=0, sticky='nw')
        spb_smoothing_spline_degree.grid(row=5, column=1, sticky='nw')                                    
        lbl_smoothing_scale_factor.grid(row=6, column=0, sticky='nw')
        spb_smoothing_scale_factor.grid(row=6, column=1, sticky='nw')
        

        
    def __confirm_callback(self):
        """
        Store all settings in the params and project file.
        """
        # Global settings
        dtrack_params["options.video.directory"] = self.__stv_video_directory.get()
        
        dtrack_params["options.autocalibration.fix_k1"] = self.__blv_fix_k1.get()
        dtrack_params["options.autocalibration.fix_k2"] = self.__blv_fix_k2.get()
        dtrack_params["options.autocalibration.fix_k3"] = self.__blv_fix_k3.get()
        dtrack_params["options.autocalibration.fix_tangential"] = self.__blv_fix_tangential.get()
        dtrack_params["options.autocalibration.show_meta_text"] = self.__blv_show_meta_text.get()
        
        dtrack_params["options.autotracker.track_point"] = self.__stv_dtrack_track_point.get()
        dtrack_params["options.autotracker.cv_backend"] = self.__stv_cv_backend.get()
        dtrack_params["options.autotracker.bg_computation_method"] = self.__stv_bg_computation_method.get()
        dtrack_params["options.autotracker.bg_sample_size"] = int(self.__stv_bg_sample_size.get())
        dtrack_params["options.autotracker.track_interval"] = int(self.__stv_track_interval.get())
        dtrack_params["options.autotracker.remember_roi"] = self.__blv_remember_roi.get()
        dtrack_params["options.autotracker.show_roi"] = self.__blv_show_roi.get()

        dtrack_params["options.processing.plot_grid"] = self.__blv_plot_grid.get()
        dtrack_params["options.processing.include_legend"] = self.__blv_include_legend.get()
        dtrack_params["options.processing.filename"] = self.__stv_plot_filename.get()
        dtrack_params["options.processing.filetype"] = self.__stv_plot_filetype.get()
        dtrack_params["options.processing.zero"] = self.__blv_plot_zero_tracks.get()
        dtrack_params["options.processing.smoothing_spline_degree"] = int(self.__stv_smoothing_spline_degree.get())
        dtrack_params["options.processing.smoothing_scale_factor"] = float(self.__stv_smoothing_scale_factor.get())

        self.destroy()

    def __select_video_directory_callback(self):
        """
        Callback for selecting the video directory.
        """

        dirname = filedialog.askdirectory()
        if dirname == ():
            # User cancelled
            return
    
        self.__stv_video_directory.set(dirname)





