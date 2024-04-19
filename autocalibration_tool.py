import tkinter as tk
from tkinter import messagebox, filedialog, dialog

from project import project_file

class AutocalibrationTool(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.title('Autocalibration tool')
        self.minsize(500, 100)


        # Want to allow user to select either the index of a frame in their
        # calibration video or give an image file. Defualt to using the frame.
        self.__extrinsic_frame = 0
        self.__extrinsic_filepath = ""

        self.__stv_extrinsic_calibration_image = tk.StringVar()
        
        self.__lbl_calib_video = tk.Label(
            self,
            text="Calibration video:",
            anchor='e')
        self.__lbl_ext_frame = tk.Label(
            self,
            text="Extrinsic (Ground) image:",
            anchor='e')
        self.__lbl_calib_video_path = tk.Label(
            self,                                               
            text=project_file["calibration_video"],
            relief='sunken',
            anchor='w')
        self.__lbl_ext_calib_frame = tk.Label(
            self,
            textvariable=self.__stv_extrinsic_calibration_image,
            relief='sunken',
            anchor='w')
        
        self.__lbf_extrinsic_selection = tk.LabelFrame(
            self,
            text="Select extrinsic frame")
        
        self.__lbl_or = tk.Label(self.__lbf_extrinsic_selection, text=' OR ')
        self.__btn_select_image = tk.Button(self.__lbf_extrinsic_selection,
                                            text="Select image file")
        self.__btn_select_frame = tk.Button(self.__lbf_extrinsic_selection,
                                            text="Select video frame")
        self.__btn_generate = tk.Button(self,
                                        text="Generate!")


        # Window geometry
        n_columns = 3
        n_rows=3
        for i in range(n_rows):
            for j in range(n_columns):
                self.rowconfigure(i, weight=1)
                self.columnconfigure(j, weight=1)

        # Extrinsic frame selector labelframe geometry
        n_columns = 3
        n_rows = 1
        for i in range(n_rows):
            for j in range(n_columns):
                self.__lbf_extrinsic_selection.rowconfigure(i, weight=1)
                self.__lbf_extrinsic_selection.columnconfigure(j, weight=1)

        # Window layout
        self.__lbl_calib_video.grid(row=0, column=0, sticky='nesw')
        self.__lbl_calib_video_path.grid(row=0, column=1, columnspan=2, sticky='nesw')

        self.__lbl_ext_frame.grid(row=2, column=0, sticky='nesw')
        self.__lbl_ext_calib_frame.grid(row=2, column=1, columnspan=2, sticky='nesw')

        self.__lbf_extrinsic_selection.grid(row=3, column=0, columnspan=2, sticky='nesw')
        self.__btn_generate.grid(row=3, column=2, sticky='nesw', padx=10, pady=10)

        # Extrinsic frame selector layout
        self.__btn_select_frame.grid(row=0, column=0, sticky='nesw')
        self.__lbl_or.grid(row=0, column=1, sticky='ew')
        self.__btn_select_image.grid(row=0, column=2, sticky='nesw')
        
        self.__update_ext_calibration_label()

    def __update_ext_calibration_label(self):
        # If an image filepath has been given, display
        if not self.__extrinsic_filepath == "":
            text = self.__extrinsic_filepath
        else:
            text = "Frame {}".format(str(self.__extrinsic_frame))
            
        self.__stv_extrinsic_calibration_image.set(text)
        

    def __select_extrinsic_frame(self):
        """
        Spawn a simple dialog box to get the integer frame for extrinsic 
        calibration from the user.
        """

        