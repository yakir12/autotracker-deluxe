import shutil
import tkinter as tk
from tkinter import filedialog

from dtrack_params import dtrack_params
from project import project_file

class SelectorFrame(tk.Frame):
    def __init__(self, parent, title='', default='', **kwargs):
        super().__init__(parent, **kwargs)

        self.__parent = parent

        # Allows all children to be resized
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.__lbf_content = tk.LabelFrame(self, text=title)
        self.__btn_select = tk.Button(self.__lbf_content,
                                      text='Select',
                                      command=self.__getfile)
        self.__ent_current = tk.Entry(self.__lbf_content)
        self.__filepath = default
        self.__update_file_entry()

        # Manage geometry
        entry_columns=3
        for i in range(entry_columns):
            self.__lbf_content.columnconfigure(i, weight=1)

        # Pack into main frame
        self.__lbf_content.grid(row=0, column=0, sticky='nesw')
        
        # Pack into Labelframe
        self.__ent_current.grid(row=0, column=0, columnspan=3, sticky='ew')
        self.__btn_select.grid(row=0, column=4, sticky='e')


    def __getfile(self):
        filename = filedialog.askopenfilename(title='Select file', 
                                              filetypes=[('All files (Video)', '*.*')],
                                              initialdir=dtrack_params["options.video.directory"],
                                              parent=self.__parent)
        
        if filename == ():
            # User cancelled
            return
        
        self.__filepath = filename
        self.__update_file_entry()
    
    def __update_file_entry(self):
        self.__ent_current.delete(0, tk.END)
        self.__ent_current.insert(0, self.__filepath)

    def get_entry_information(self):
        return self.__ent_current.get()


class VideoSelector(tk.Toplevel):
    """
    Video selection window object. To be created where the user wants to
    select a calibration or tracking video for the project.
    """
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)

        self.title("Select video files")
        self.minsize(600, 150)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.__content = tk.Frame(self)
        
        n_rows = 3
        for i in range(n_rows):
            self.__content.rowconfigure(i, weight=1)

        self.__content.columnconfigure(0, weight=1)


        # Create widgets
        self.__slt_calibration_video = SelectorFrame(self.__content,
                                                     title='Calibration video',
                                                     default=project_file['calibration_video'])
        self.__slt_tracking_video = SelectorFrame(self.__content,
                                                  title='Video for tracking',
                                                  default=project_file['tracking_video'])
        
        self.__frm_button_frame = tk.Frame(self)
        self.__btn_confirm = tk.Button(self.__frm_button_frame,
                                       text='Confirm',
                                       command=self.__confirm_callback)
        self.__btn_cancel = tk.Button(self.__frm_button_frame,
                                      text='Cancel',
                                      command=self.__cancel_callback)
        
        self.__blv_copy = tk.BooleanVar(value=False)
        self.__chk_copy_local = tk.Checkbutton(self.__frm_button_frame,
                                               text='Copy video files locally?',
                                               variable=self.__blv_copy)

        self.__content.grid(row=0, column=0, sticky='nesw')

        self.__slt_calibration_video.grid(row=0, column=0, sticky='nesw')
        self.__slt_tracking_video.grid(row=1, column=0, sticky='nesw')
        self.__frm_button_frame.grid(row=2, column=0, sticky='nesw')
        
        self.__btn_confirm.grid(column=0, row=0, sticky='ne')
        self.__btn_cancel.grid(column=1, row=0, sticky='ne')
        self.__chk_copy_local.grid(column=3, row=0, sticky='ne')

        
    def __confirm_callback(self):
        # Store parameters and destory window
        calib_video = self.__slt_calibration_video.get_entry_information()
        track_video = self.__slt_tracking_video.get_entry_information()

        if self.__blv_copy.get():
            # Preserve filename
            calib_video_filename = calib_video.split("/")[-1]
            track_video_filename = track_video.split("/")[-1]


            # Given that we have the project file record, why rename these at 
            # all? We don't have to assume video filenames anymore
            local_calib_name =\
                 dtrack_params['project_directory'] + "/" + calib_video_filename
                      
            local_track_name =\
                  dtrack_params['project_directory'] + "/" + track_video_filename

            # Copy video files locally
            try:
                shutil.copy(calib_video, local_calib_name)    
            except shutil.SameFileError:
                pass

            try:
                shutil.copy(track_video, local_track_name)
            except shutil.SameFileError:
                pass

            print("Video files copied to:")
            print(local_calib_name)
            print(local_track_name)

            # Insert original filenames into project file. These are only really 
            # stored so that there's a record of what the original filepath was.
            project_file["original_calibration_video"] = calib_video
            project_file["original_tracking_video"] = track_video

            # Overwrite so that project file points to local copies
            calib_video = local_calib_name
            track_video = local_track_name
        
        project_file["calibration_video"] = calib_video
        project_file["tracking_video"] = track_video

        self.destroy()

    def __cancel_callback(self):
        # Destroy window, do not save videofile settings
        self.destroy()
        
        