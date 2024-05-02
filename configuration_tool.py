import tkinter as tk
from tkinter import ttk

from dtrack_params import dtrack_params
from project import project_file

class ConfigurationTool(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        # Configure window
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.minsize(500,300)

        self.__ntb_notebook = ttk.Notebook(self)
        
        self.__frm_project_options = tk.Frame(self.__ntb_notebook)
        self.__frm_dtrack_options = tk.Frame(self.__ntb_notebook)

        self.__ntb_notebook.grid(column=0, row=0, sticky='nesw')
        self.__frm_project_options.grid(row=0, column=0, sticky='nesw')
        self.__frm_dtrack_options.grid(row=0, column=0, sticky='nesw')
        
        self.__ntb_notebook.add(self.__frm_project_options, text='Project')
        self.__ntb_notebook.add(self.__frm_dtrack_options, text='DTrack2') 

        # Definition and arrangement of each tab is contained in these functions
        # to try to organise the source code a little.
        self.__arrange_project_options_pane()
        self.__arrange_dtrack_options_pane()

    def __config_rows_and_cols(self, 
                               widget: tk.Frame, 
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

        

    def __arrange_project_options_pane(self):
        """
        Define all widgets for the project options panel and 'grid' them for display.
        """

        n_rows = 1
        n_cols = 1

        for row in range(n_rows):
            for col in range(n_cols):
                self.__frm_project_options.rowconfigure(row, weight=1)
                self.__frm_project_options.columnconfigure(col, weight=1)

        # Project options info-string
        str = "This pane allows you to configure options for the current project."
        lbl_project_option_info = tk.Label(self.__frm_project_options,
                                                  text=str, 
                                                  anchor='center')
        lbl_project_option_info.grid(column=0, row=0, sticky='new')      


        # Tracker options
        lbf_tracker_options = tk.LabelFrame(self.__frm_project_options,
                                                   text="Autotracker")
        
        
        
        lbl_track_point_selection = tk.Label(lbf_tracker_options,
                                             text="Autotracker target")
        cmb_track_point_selection = ttk.Combobox(lbf_tracker_options,
                                                 values=["centre-of-mass",
                                                         "centre-of-bounding-box"])     
        
        lbl_track_point_selection.grid()

    def __arrange_dtrack_options_pane(self):
        """
        Define all widgets for the DTrack2 options panel and 'grid' them for display.
        """
        n_rows = 1
        n_cols = 1

        for row in range(n_rows):
            for col in range(n_cols):
                self.__frm_dtrack_options.rowconfigure(row, weight=1)
                self.__frm_dtrack_options.columnconfigure(col, weight=1)

        str = "This pane allows you to configure global options for DTrack 2"
        self.__lbl_dtrack_option_info = tk.Label(self.__frm_dtrack_options, 
                                                 text=str, 
                                                 anchor='center')
        self.__lbl_dtrack_option_info.grid(column=0,row=0, sticky='new')

