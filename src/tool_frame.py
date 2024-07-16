"""
tool_frame.py

Provides a Tkinter UI for selecting and running different tools.
"""

import tkinter as tk
import webbrowser
import os

from wrapped_label import WrappedLabelFrame

from video_selector import VideoSelector
from chessboard_selector import ChessboardSelector
from autotrack import autotracker
from old_calibration import calib
from calibration_manager import CalibrationManager
from track_processing import calibrate_and_smooth_tracks
from configuration_tool import ConfigurationTool


class ToolFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.__labelframe = tk.LabelFrame(self, text='Tools')
        self.__frm_left_button_frame = tk.Frame(self.__labelframe)
        self.__frm_right_button_frame = tk.Frame(self.__labelframe)
    
        self.__btn_run = tk.Button(self.__frm_left_button_frame, 
                                   text='Run',
                                   command=self.__run_callback)
        self.__btn_quit = tk.Button(self.__frm_left_button_frame,
                                    text='Quit',
                                    command=self.winfo_toplevel().destroy)  
        self.__btn_configure = tk.Button(self.__frm_right_button_frame,
                                         text='Options',
                                         command=self.__run_configuration)
        self.__btn_help = tk.Button(self.__frm_right_button_frame,
                                    text='Help',
                                    command=lambda: webbrowser.open(
                                        os.path.join("documentation", "main.html")
                                    ))

        # Tool radio buttons
        self.__int_option = tk.IntVar(self, 1)
        self.__rbn_select_videos =\
              tk.Radiobutton(self.__labelframe,
                             text='1. Choose video files',
                             variable=self.__int_option,
                             value=1,
                             command=self.__update_info)
        self.__rbn_configure_calibration =\
              tk.Radiobutton(self.__labelframe,
                             text='2. Configure calibration board',
                             variable=self.__int_option,
                             value=2,
                             command=self.__update_info)
        self.__rbn_compute_calibration =\
              tk.Radiobutton(self.__labelframe,
                             text='3. Calibration manager',
                             variable=self.__int_option,
                             value=3,
                             command=self.__update_info)
        self.__rbn_autotracker =\
              tk.Radiobutton(self.__labelframe,
                             text='4. Run autotracker',
                             variable=self.__int_option,
                             value=4,
                             command=self.__update_info)
        self.__rbn_process_tracks =\
              tk.Radiobutton(self.__labelframe,
                             text='5. Process tracks (zero, calibrate, smooth)',
                             variable=self.__int_option,
                             value=5,
                             command=self.__update_info)
        
        self.__txtvar_information = tk.StringVar()
        self.__lfm_information = tk.LabelFrame(self.__labelframe,
                                               text="Information", 
                                               relief='sunken')
        self.__lfm_information.columnconfigure(0, weight=1)
        self.__lfm_information.rowconfigure(0, weight=1)
        self.__wlf_information = WrappedLabelFrame(self.__lfm_information)
                                                                                      

        self.__update_info() # Populate the information label.
                
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        n_rows = 6
        n_columns = 2
        for i in range(n_rows):
            for j in range(n_columns):
                self.__labelframe.rowconfigure(i, weight=1)
                self.__labelframe.columnconfigure(j, weight=1)
            
        
        # Contained in 'self'
        self.__labelframe.grid(column=0, row=0, sticky='nesw')

        # Contained in self.__labelframe
        self.__rbn_select_videos.grid(column=0, row=0, sticky='nw')
        self.__rbn_configure_calibration.grid(column=0, row=1, sticky='nw') 
        self.__rbn_compute_calibration.grid(column=0, row=2, sticky='nw') 
        self.__rbn_autotracker.grid(column=0, row=3, sticky='nw') 
        self.__rbn_process_tracks.grid(column=0, row=4, sticky='nw') 
        self.__frm_left_button_frame.grid(column=0, row=5, sticky='nw')
        self.__frm_right_button_frame.grid(column=1, row=5, sticky='ne')
        self.__lfm_information.grid(column=1, row=0, rowspan=5, sticky='nesw', padx=(0,10))
        self.__wlf_information.grid(column=0, row=0, sticky='nesw')

        # Contained in self.__frm_right_button_frame
        self.__btn_run.grid(column=0, row=0, sticky='nw', pady=(5,5), padx=(10,0))
        self.__btn_quit.grid(column=1, row=0, sticky='nw', pady=(5,5))

        # Contained in self.__frm_left_button_frame
        self.__btn_configure.grid(column=2, row=0, sticky='ne', pady=(5,5), padx=(0,10))
        self.__btn_help.grid(column=1, row=0, sticky='ne', pady=(5,5), padx=(0,10))

    def __update_info(self):
        """
        Callback tied to Radiobutton options in order to display some useful
        information when the option is selected.
        """

        # Get the current radiobutton selection to determine which text to 
        # display
        var = self.__int_option.get()
        
        if var == 1:
            self.__wlf_information.set_text(
                "Select the video files you wish to use for tracking and" +
                " calibration.\n\nYou only need to do this once and doing this" +
                " multiple times will overwrite your old video selection.\n\n" +
                "If you are going to import an existing calibration file, you do not need " +
                "to select a calibration video.") 
            
        elif var == 2:
            self.__wlf_information.set_text(
                "Specify the calibration board dimensions and square size."
            )
        elif var == 3:
            self.__wlf_information.set_text(
                "(NEW) Run the calibration manager. This will allow you to generate a" +\
                " calibration file from your calibration video or import an existing"+\
                " calibration file from another project."
            )
        elif var == 4:
            self.__wlf_information.set_text(
                "Run the autotracker. Produces a file 'raw_tracks.csv' in the" + 
                " session directory. If this file already exists, new tracks will" + 
                " be appended to the end." 
            )
        elif var == 5:
            self.__wlf_information.set_text(
                "Run post-processing on tracks to zero, calibrate (undistort and" +
                " convert to real-world disance), and smooth the tracks. Each" + 
                " stage of post-processing produces a CSV file which you can use" +
                " to produce your own plots. This option will produce a rudimentary" +
                " plot so you can check that it's working correctly. \n\n"
                "(NEW) Calibration is now enabled!"
                )            
            self.__txtvar_information.set("")
        
    def __run_callback(self):
        """
        Callback tied to 'Run' button. Will select what tool to run based
        on the current radiobutton selection.
        """
        var = self.__int_option.get()

        if var == 1:
            # Select videos
            selector_window = VideoSelector(self)
            selector_window.mainloop()
        elif var == 2:
            chessboard_selector = ChessboardSelector(self)
            chessboard_selector.mainloop()
        elif var == 3:
            #calib()
            calib_manager = CalibrationManager(self)
            calib_manager.mainloop()
        elif var == 4:
            autotracker()
        elif var == 5:
            calibrate_and_smooth_tracks()

    def __run_configuration(self):
        """
        Button callback to launch the configuration tool
        """
        config_tool = ConfigurationTool(self)    
        config_tool.mainloop()

