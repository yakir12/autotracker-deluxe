"""
project.py

Provides a (project-level) parameter file passthrough class. This provides 
an object via which a class can interact with the project file with some
safeguards (key checks and default values).

The module also defines an instance of this class 'project_file' which you can
import to other modules.
"""

import json
import os

from dtrack_params import dtrack_params

class ProjectFilePassthrough():
    """
    Class to act as an interaction layer with project files. This abstracts
    file interaction and allows you to use the project file as if it were
    a dictionary.
    """
    def __init__(self):
        self.__fname = dtrack_params["project_file"]

        self.__valid_keys = ["calibration_video", 
                             "tracking_video",
                             "original_tracking_video",
                             "original_calibration_video",
                             "chessboard_rows",
                             "chessboard_columns",
                             "chessboard_square_size",
                             "chessboard_size",
                             "calibration_cache",
                             "calibration_file",
                             "track_fps"]
        
        # Set some sensible defaults for things which may have
        # default values.
        self.__defaults = dict()
        self.__defaults["chessboard_rows"] = 6
        self.__defaults["chessboard_columns"] = 9
        self.__defaults["chessboard_size"] = (self.__defaults["chessboard_columns"] - 1,
                                              self.__defaults["chessboard_rows"] - 1)
        self.__defaults["chessboard_square_size"] = 39
        self.__defaults["calibration_cache"] =\
              os.path.join(dtrack_params['project_directory'], 'calibration_cache')
        self.__defaults["calibration_file"] =\
              os.path.join(self.__defaults["calibration_cache"], "calibration.dt2c")
        self.__defaults["options.autotracker.track_point"] =\
              dtrack_params["options.autotracker.track_point"]
        self.__defaults["track_fps"] = -1

    def __getitem__(self,key):
        self.refresh()

        assert(key in self.__valid_keys)

        with open(self.__fname, "r") as f:
                project_file = json.load(f)

        # Feels disgusting but return the item if it's been set,
        # otherwise try the defaults dictionary. If no default is set
        # then return the empty string.
        try:
            item = project_file[key]
        except KeyError:
            try:
                item = self.__defaults[key]
                self.__setitem__(key, item) # Add to toplevel representation
            except KeyError:
                item = ""

        return item
    
    def __setitem__(self, key, value):
        self.refresh()

        # Failing this assertion indicates programmer error.
        assert(key in self.__valid_keys)

        with open(self.__fname, "r") as f:
            project_file = json.load(f)

        project_file[key] = value
        
        with open(self.__fname, "w") as f:
            json.dump(project_file, f, indent=2)

    def refresh(self):
        """
        If called, this will update the file that the project file interaction
        object points to. This may be used in cases where the project is changed
        after the project_file interaction object has been imported.

        Alternatively you could instantiate a new ProjectFilePassthrough() object
        after dtrack_params had been updated.
        """
        self.__fname = dtrack_params["project_file"]

        # Refresh all default values which depend on the information from dtrack_params
        self.__defaults["calibration_cache"] =\
              os.path.join(dtrack_params['project_directory'], 'calibration_cache')
        self.__defaults["calibration_file"] =\
              os.path.join(self.__defaults["calibration_cache"], "calibration.dt2c")
    
    def name(self):
        return self.__fname
        
        
   
# Insantiation of project file interaction object.
project_file = ProjectFilePassthrough()