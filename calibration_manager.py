import os
import shutil

import tkinter as tk
from tkinter import filedialog, messagebox
from enum import Enum

import autocalibration
import calibration as calib

from autocalibration_tool import AutocalibrationTool

from dtrack_params import dtrack_params
from project import project_file

class CalibStatus(Enum):
    EXISTS = 1
    CORRUPT = 2
    NOT_FOUND = 0

class CalibrationManager(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Calibration manager")

        self.__frm_content = tk.Frame(self)
        
        self.__stv_calib_status = tk.StringVar()
        self.__lbl_calib_status = tk.Label(self.__frm_content,
                                           textvariable=self.__stv_calib_status)
        
        self.__btn_generate_calibration =\
            tk.Button(self.__frm_content,  
                      text="Generate new calibration",
                      command=self.__launch_autocalibration_tool)

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


        self.__check_for_calibration()
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
            self.__stv_calib_status.set("This project has a calibration file!")
            self.__lbl_calib_status.configure(fg='#007d02')
        elif self.__calib_status == CalibStatus.CORRUPT:
            self.__stv_calib_status.set("Selected calibration corrupt! You should generate a new calibration.")
            self.__lbl_calib_status.configure(fg='#00457d')
        
    def __launch_autocalibration_tool(self):
        """
        Generate a new calibration using the autocalibration tool.
        """

        # Check whether the calibration video has been set   
        if project_file["calibration_video"] == '':
            msg = "No calibration video has been set for this project.\n\nPlease run" +\
                  " Tool #1 from the main window (Choose video files) and set the" +\
                  " desired calibration video."
            messagebox.showerror("No calibration video!",
                                 message=msg)
            
            return

        # Create new window which manages the autocalibration
        auto_calibration = AutocalibrationTool(self)
        auto_calibration.mainloop()


        #self.__check_for_calibration()
        

    def __import_calibration(self):
        """
        Import an existing calibration from a different project.
        """
        #
        # Calibration file import
        #
        filepath_for_import = filedialog.askopenfilename(
            title="Select calibration file",
            filetypes=[("DTrack2 Calibration Files", ".dt2c")])
        
        if filepath_for_import == '':
            # User cancelled.
            return

        try:
            shutil.copy(filepath_for_import, project_file["calibration_file"])
        except shutil.SameFileError:
            msg = "You tried to import the calibration file associated with this " +\
                  "project. (i.e. You're trying to copy a file to itself.) Make "+\
                  "sure the source for the calibration file isn not your current project!"
            
            messagebox.showerror(title="Same file error!",
                                 message=msg)

        # Check file exists in the correct place and verify file integrity.
        self.__check_for_calibration()

    def __check_calibration(self):
        """
        Check the calibration visually and display some estimated distances for
        the user.
        """

        # Check to see if we have an example extrinsic frame to use
        if not os.path.exists(os.path.join(project_file["calibration_cache"], 
                                           'extrinsic', 
                                           '000.png')):
            msg = "Checking the calibration requires an example image where the " +\
                  "chessboard is placed on the ground. If you imported a previous " +\
                  "calibration, then you need to provide this image (a file "+\
                  "selection dialog will open when you close this window)."
            
            messagebox.showwarning(title="No example frame found!",
                                   message=msg)

            ext_image_path = filedialog.askopenfilename(
                title="Select calibration file",
                filetypes=[("PNG image files", ".png")]
            )
        
            if ext_image_path == '':
                # User cancelled.
                return
            

        #
        # STUB, need to be able to generate calibrations before we can test them.
        #
    

    
    def __check_for_calibration(self):
        """
        Check

        1. Does calibration cache exist?
        2. Does calibration file exist?
        3. If so, does it correspond to a Calibration object when loaded?

        This will create a calibration cache directory if one does not already 
        exist.
        """

        directory = project_file["calibration_cache"]
        filepath = project_file["calibration_file"]

        if os.path.exists(directory):
            if os.path.exists(filepath):
                if not calib.verify_calibration(filepath):
                    print("Warning! Calibration file found but file structure" +
                          " is not recognised. File may be corrupt or malicious." +
                          " Generate a new calibration.")
                    self.__calib_status = CalibStatus.CORRUPT
                else:
                    self.__calib_status = CalibStatus.EXISTS
            else:
                self.__calib_status = CalibStatus.NOT_FOUND
        else:
            print("Calibration cache not found, creating at {}".format(directory))
            os.mkdir(directory)
            self.__calib_status = CalibStatus.NOT_FOUND
        
        self.__update_calib_message()


