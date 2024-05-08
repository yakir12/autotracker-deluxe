import tkinter as tk
import os
from tkinter import filedialog, messagebox
from dtrack_params import dtrack_params

import json

class ProjectFrame(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.__labelframe = tk.LabelFrame(self, text='Project information')
        self.__label = tk.Label(self.__labelframe, 
                                text='Project directory: ')
        self.__btn_select =  tk.Button(self.__labelframe,
                                       text='Select project',
                                       command=self.__select_callback)
        self.__btn_new = tk.Button(self.__labelframe,
                                   text='New project',
                                   command=self.__new_callback)
        self.__ent_project = tk.Entry(self.__labelframe, width=80)
        
        # If the user has previously worked on a project and that project
        # still exists, populate the project entry with the relevant info.
        cached_project = dtrack_params['project_directory']
        if not (cached_project == None):
            if os.path.exists(cached_project):
                self.__update_project_entry()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        n_cols = 5
        for i in range(n_cols):
            if (i == 1) or (i==2) or (i==3):
                self.__labelframe.columnconfigure(i, weight=1)
                continue
            self.__labelframe.columnconfigure(i, weight=0)

        self.__labelframe.rowconfigure(0,weight=1)

        self.__labelframe.grid(row=0, column=0,sticky='ew')
        self.__label.grid(row=0, column=0, sticky='w')
        self.__ent_project.grid(row=0, column=1, columnspan=2, sticky='ew')               
        self.__btn_select.grid(row=0, column=3, sticky='e')
        self.__btn_new.grid(row=0, column=4, sticky='e')

    def __select_callback(self):
        """
        Spawn a file dialog when project selection window is shown.
        """
        directory = filedialog.askdirectory(
            initialdir = str(os.getcwd()),
            title = "Select project directory"
        )

        # User cancelled
        if directory == ():
            return

        if not os.path.exists(directory):
            directory = ""
            print("Error: selected directory does not exist.")

        # Check to see if a project file exists
        proj_file_name = os.path.basename(directory) + ".dt2p" 
        proj_file_path = os.path.join(directory, proj_file_name)
        if not os.path.isfile(proj_file_path):
            # Check that path doesn't already exist
            confirm_message =\
              "A project file (*.dt2p) does not exist in this directory." +\
              "Do you wish to create one?\n\n" +\
              "(This could either be because the selected directory is not a" +\
              " dtrack2 project or because the project was created before the" +\
              " introduction of project files.)"

            confirm = messagebox.askokcancel(title="Confirm directory", 
                                             message=confirm_message,
                                             icon='question')
            if confirm:
                with open(proj_file_path, "w") as f:
                    project = dict()
                    json.dump(project, f, indent=2)
            else:
                # If the user didn't create a project file, abort project 
                # selection
                return

        # Update dtrack_params file
        self.__update_project_params(directory)

        # Update the Entry widget to display the current project
        self.__update_project_entry()


    def __new_callback(self):
        """
        Spawn a file dialog to select the parent directory.
        """
        full_path = filedialog.asksaveasfilename(
            initialdir = str(os.getcwd()),
            title = "Save as",
            filetypes=[("Dung Track 2 Project (JSON)", "*.dt2p")]
        )

        # User cancelled 
        if full_path == ():
            return

        # Check that the user really wants to create the project
        parent = full_path.split(".")[0]
        confirm_message =\
              "A project directory will be created\n{}\n\n".format(parent)+\
              "Do you wish to proceed?"

        confirm = messagebox.askokcancel(title="Confirm directory", 
                                         message=confirm_message,
                                         icon='question')
        
        if confirm:
            # Create parent directory
            os.mkdir(parent)
            filename = os.path.basename(full_path)
            full_path = os.path.join(parent, filename)

            # Create project (JSON) file
            with open(full_path, "w") as f:
                project = dict()
                json.dump(project, f, indent=2)

        self.__update_project_params(parent)
        self.__update_project_entry()

    def get_session(self):
        return self.__ent_project.get()
    
    def __update_project_entry(self):
        """
        Update the contents of the current project entry to match the 
        parameters stored in dtrack_params.
        """
        self.__ent_project.delete(0, tk.END)
        self.__ent_project.insert(0, dtrack_params['project_directory'])

    def __update_project_params(self, project_directory):
        """
        Given a project directory, set the relevant dtrack_params to that project
        and the associated project file. This should ensure the value is correct
        for any sub-routines.
        
        Note: it is assumed that the project file has been created which occurs
        in the new project callback.

        :param project_directory:
        """
        dtrack_params["project_directory"] = project_directory
        filename = os.path.basename(project_directory) + ".dt2p"
        filepath = os.path.join(project_directory,filename)
        dtrack_params["project_file"] = filepath

