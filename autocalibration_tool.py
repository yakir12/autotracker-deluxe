import shutil
import os
import tkinter as tk
from tkinter import messagebox, filedialog, dialog

from project import project_file

import autocalibration as ac

class AutocalibrationTool(tk.Toplevel):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.title('Autocalibration tool')
        self.minsize(500, 100)


        # Want to allow user to select either the index of a frame in their
        # calibration video or give an image file. Defualt to using the frame.
        self.__extrinsic_frame = 0
        self.__extrinsic_filepath = ""
        self.__extrinsic_frame_set = False

        self.__stv_extrinsic_calibration_image = tk.StringVar()
        self.__stv_N_frames = tk.StringVar()
        self.__stv_N_frames.set("30")
        
        self.__lbl_calib_video = tk.Label(
            self,
            text="Calibration video:",
            anchor='e')
        self.__lbl_ext_frame = tk.Label(
            self,
            text="Extrinsic (Ground) image:",
            anchor='e')
        
        self.__lbl_N_frames = tk.Label(
            self,
            text="N calibration frames: ",
            anchor='e'
        )

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
        
        # Entry to allow the user to configure the number of frames to use for
        # calibration. Validation function called on each keypress.
        self.__ent_N_frames = tk.Entry(
            self,
            textvariable=self.__stv_N_frames,
            validatecommand=(
                self.register(self.__N_frame_entry_validation), '%P'))
        
        self.__lbf_metadata = tk.LabelFrame(
            self,
            text="Metadata")
        
        self.__txt_metadata = tk.Text(
            self.__lbf_metadata,
            width=60,
            height=10, 
            wrap='word')
        
        metadata_instructions =\
            "Imagine you are using this calibration file without access to the" +\
            " original video. You may want to know date, time, location, experiment, etc. \n\n" +\
            "Include that information here! It will be encoded in the calibration file."

        self.__txt_metadata.insert(tk.END, metadata_instructions)
        
        self.__lbf_extrinsic_selection = tk.LabelFrame(
            self,
            text="Select extrinsic frame")
        
        self.__lbl_or = tk.Label(self.__lbf_extrinsic_selection, text=' OR ')
        self.__btn_select_image = tk.Button(self.__lbf_extrinsic_selection,
                                            text="Select image file",
                                            command=self.__select_extrinsic_frame_from_file)
        self.__btn_select_frame = tk.Button(self.__lbf_extrinsic_selection,
                                            text="Select video frame",
                                            command=self.__select_extrinsic_frame_from_video)
        self.__btn_generate = tk.Button(self,
                                        text="Generate!",
                                        command=self.__generate_calibration)

        # Window geometry
        n_columns = 3
        n_rows = 5
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

        self.__lbl_N_frames.grid(row=1, column=0, sticky='nesw')
        self.__ent_N_frames.grid(row=1, column=1, columnspan=2, sticky='nesw')

        self.__lbf_metadata.grid(row=2, column=0, columnspan=3, sticky='nesw')
        
        self.__lbl_ext_frame.grid(row=3, column=0, sticky='nesw')
        self.__lbl_ext_calib_frame.grid(row=3, column=1, columnspan=2, sticky='nesw')

        self.__lbf_extrinsic_selection.grid(row=4, column=0, columnspan=2, sticky='nesw')
        self.__btn_generate.grid(row=4, column=2, sticky='nesw', padx=10, pady=10)

        # Extrinsic frame selector layout
        self.__btn_select_frame.grid(row=0, column=0, sticky='nesw')
        self.__lbl_or.grid(row=0, column=1, sticky='ew')
        self.__btn_select_image.grid(row=0, column=2, sticky='nesw')

        # Metadata frame layout
        self.__lbf_metadata.columnconfigure(0, weight=1)
        self.__lbf_metadata.rowconfigure(0, weight=1)
        self.__txt_metadata.grid(row=0, column=0, sticky='nesw')
        
        self.__update_ext_calibration_label()

    def __N_frame_entry_validation(self, input):
        """
        Validate input to the N frame entry (number of frames to use for 
        intrinsic calibration).

        Should only accept integers

        :param input: The content of the entry at the time the function was called.
        """
        if input.isdigit() or input == '':
            return True
        
        return False


    def __update_ext_calibration_label(self):
        if self.__extrinsic_frame_set:
            # If frame is set, make text green
            self.__lbl_ext_calib_frame.configure(fg='#007d02')

            # If a filepath has been set, use that.
            if not self.__extrinsic_filepath == "":
                text = self.__extrinsic_filepath
            else:
                # Otherwise show which frame the user selected
                text = "Frame {}".format(str(self.__extrinsic_frame))
        else:
            # If no frame has been selected, show text in red.
            text = 'No extrinsic frame set!'
            self.__lbl_ext_calib_frame.configure(fg='#eb3a34')

        self.__stv_extrinsic_calibration_image.set(text)
        

    def __select_extrinsic_frame_from_video(self):
        """
        Spawn an OpenCV window with the calibration video.

        Allows user to select an extrinsic calibration frame from the calibration video
        """
        
        frame_was_set, frame_idx = ac.select_extrinsic_frame(
            project_file['calibration_video'],
            project_file['calibration_cache'],
            project_file['chessboard_size'])   
        
        if frame_was_set:
            self.__extrinsic_frame = frame_idx
            self.__extrinsic_frame_set = True
            self.__update_ext_calibration_label()

    def __select_extrinsic_frame_from_file(self):
        """
        Open a file dialog to allow the user to select an extrinsic calibration
        frame from a file.
        """

        ext_image_path = filedialog.askopenfilename(
                title="Select calibration file",
                filetypes=[("PNG image files", ".png")]
            )
        
        if ext_image_path == '':
            # User cancelled
            return
        
        # Check that the corners of the chessboard can actually be found in
        # the provided image. Display the image with the corners or a message
        # if no corners could be found.
        success = ac.store_corners_from_image_file(ext_image_path,
                                                   project_file['chessboard_size'],
                                                   os.path.join(project_file['calibration_cache'],
                                                   'extrinsic'))
        
        # If no corners found or the user quit the verification stage, return
        # without doing anything.
        if not success:
            return
        
        # Try to copy file locally
        local_ext_path = os.path.join(project_file['calibration_cache'], 
                                      'extrinsic',
                                      '000.png')
    
        if not local_ext_path == ext_image_path:
            # If files are not the same, perform a copy
            shutil.copy(ext_image_path, local_ext_path)
        
        # Update the extrinsic filepath variable for display
        self.__extrinsic_filepath = ext_image_path
        self.__extrinsic_frame_set = True
        self.__update_ext_calibration_label()
        

    def __generate_calibration(self):
        """
        Generate calibration file from the calibration video and selected
        extrinsic calibration frame.
        """

        if not self.__extrinsic_frame_set:
            msg = "You need to set an extrinsic frame in order to generate" +\
                  " a calibration file."
            
            messagebox.showerror(title="No extrinsic calibration image selected",
                                 message=msg)

            # Do nothing and return    
            return
        
        N_frames = int(self.__stv_N_frames.get())
        
        cache_success =\
              ac.cache_calibration_video_frames(project_file['calibration_video'],
                                                project_file['chessboard_size'],
                                                N=N_frames,
                                                frame_cache=project_file['calibration_cache'])
        
        if not cache_success:
            msg = "Construction of the calibration cache failed. Check the" +\
                  " terminal for specific error information."
            
            messagebox.showerror(title="Cache construction failure",
                                 message=msg)
            return
        
        
        print("")

        metadata = self.__txt_metadata.get('0.0', tk.END)

        calibration_success =\
            ac.generate_calibration_from_cache(
                int(project_file['chessboard_columns']),
                int(project_file['chessboard_rows']),
                int(project_file['chessboard_square_size']),
                cache_path=project_file['calibration_cache'],
                metadata=metadata)
        
        if not calibration_success:
            msg = "Camera calibration failed. Check the terminal for specific" +\
                  " error information."
            
            messagebox.showerror(title="Calibration failed",
                                 message=msg)
            return            
        
        print("")
        print("Calibration completed successfully!")
        print("Closing autocalibration tool.")
        print("")

        self.destroy()

        

        



        
        
        
        
