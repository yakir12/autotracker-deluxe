import os
import pickle as pkl
import tkinter as tk
from tkinter import filedialog
from enum import Enum


import autocalibration
import calibration as calib

from dtrack_params import dtrack_params
from project import project_file

class CalibStatus(Enum):
    EXISTS = 1
    CORRUPT = 2
    NOT_FOUND = 0
    

class CalibrationManager(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Autocalibration tool")

        self.__calibration_filepath = ""

        self.__frm_content = tk.Frame(self)
        
        self.__stv_calib_status = tk.StringVar()
        self.__lbl_calib_status = tk.Label(self.__frm_content,
                                           textvariable=self.__stv_calib_status)
        
        self.__btn_generate_calibration =\
            tk.Button(self.__frm_content,  
                      text="Generate new calibration",
                      command=self.__lauch_autocalibration_tool)

        self.__btn_import_calibration =\
            tk.Button(self.__frm_content,  
                      text="Import existing calibration",
                      command=self.__import_calibration)
        
        self.__btn_check_calibration =\
            tk.Button(self.__frm_content,  
                      text="Check calibration",
                      command=self.__check_calibration)
        
        self.__btn_close =\
            tk.Button(self.__frm_content,  
                      text="Close",
                      command=self.destroy)        

        # Manage geometry
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)


        content_n_cols = 4
        content_n_rows = 2

        for i in range(content_n_rows):
            for j in range(content_n_cols):
                self.__frm_content.rowconfigure(i, weight=1)
                self.__frm_content.columnconfigure(j, weight=1)

        # Window population
        self.__frm_content.grid(row=0, column=0, sticky='nesw')

        # Content population
        self.__lbl_calib_status.grid(column=0, row=0, columnspan=5, sticky='nesw')
        self.__btn_generate_calibration.grid(column=0, row=1, sticky='nesw')
        self.__btn_import_calibration.grid(column=1, row=1, sticky='nesw')
        self.__btn_check_calibration.grid(column=2, row=1, sticky='nesw')
        self.__btn_close.grid(column=3, row=1, sticky='nesw')


        self.__calib_status = self.__check_for_calibration()
        self.__update_calib_message() 

    def __update_calib_message(self):
        """
        Check stored calibration status and update calibration status label
        accordingly.
        """
        if self.__calib_status == CalibStatus.NOT_FOUND:
            self.__stv_calib_status.set("No calibration selected")
            self.__lbl_calib_status.configure(fg='#eb3a34')
        elif self.__calib_status == CalibStatus.EXISTS:
            self.__stv_calib_status.set("Calibration found!")
            self.__lbl_calib_status.configure(fg='#007d02')
        elif self.__calib_status == CalibStatus.CORRUPT:
            self.__stv_calib_status.set("Selected calibration corrupt! You should generate a new calibration.")
            self.__lbl_calib_status.configure(fg='#00457d')
        
    def __lauch_autocalibration_tool(self):
        """
        Generate a new calibration using the autocalibration tool.
        """

        # Create new window which manages the autocalibration

        self.__stv_calib_status.set("New calibration generated!")
        self.__lbl_calib_status.configure(fg='#007d02')
        pass

    def __import_calibration(self):
        """
        Import an existing calibration from a different project.
        """

        # Open file dialog, copy calibration locally

        self.__stv_calib_status.set("Calibration imported!")
        self.__lbl_calib_status.configure(fg='#007d02')
        pass

    def __check_calibration(self):
        """
        Check the calibration visually and display some estimated distances for
        the user.
        """
        pass

    
    def __check_for_calibration(self):
        """
        Check

        1. Does calibration cache exist?
        2. Does calibration file exist?

        This will create a calibration cache directory if one does not already 
        exist.

        :return: CalibStatus enum
        """

        directory = project_file["calibration_cache"]
        filepath = project_file["calibration_file"]

        if os.path.exists(directory):
            if os.path.exists(filepath):
                # Attempt to load the file and check its status.
                calibration = calib.from_file(filepath)
                if not isinstance(calibration, calib.Calibration):
                    print("Warning! Calibration file found but file structure" +
                          " is not recognised. File may be corrupt or malicious." +
                          " Generate a new calibration.")
                    return CalibStatus.CORRUPT

                return CalibStatus.EXISTS
        else:
            print("Calibration cache not found, creating at {}".format(directory))
            os.mkdir(directory)

        return CalibStatus.NOT_FOUND


