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
                             "chessboard_square_size"]

    def __getitem__(self,key):
        self.refresh()

        assert(key in self.__valid_keys)

        with open(self.__fname, "r") as f:
                params = json.load(f)

        try:
            item = params[key]
        except KeyError:
            # If the item has not yet been set, then we return the 
            # empty string.
            item = ""

        return item
    
    def __setitem__(self, key, value):
        self.refresh()

        # Failing this assertion indicates programmer error.
        assert(key in self.__valid_keys)

        with open(self.__fname, "r") as f:
            params = json.load(f)

        params[key] = value
        
        with open(self.__fname, "w") as f:
            json.dump(params, f)

    def refresh(self):
        """
        If called, this will update the file that the project file interaction
        object points to. This may be used in cases where the project is changed
        after the project_file interaction object has been imported.

        Alternatively you could instantiate a new ProjectFilePassthrough() object
        after dtrack_params had been updated.
        """
        self.__fname = dtrack_params["project_file"]
    
    def name(self):
        return self.__fname
        
        
   
# Insantiation of project file interaction object.
project_file = ProjectFilePassthrough()