"""
calibration.py

Provides a class structure for camera calibration information and
utilities to save and load calibration objects.
"""

import pickle

class Calibration():
    """
    Basic class to hold calibration information
    """
    def __init__(self,
                 matrix=None,
                 distortion=None,
                 opt_matrix=None,
                 rvecs=None,
                 tvecs=None,
                 reprojection_error=None,
                 perspective_transform=None,
                 scale=None):
        self.camera_matrix = matrix
        self.distortion = distortion
        self.opt_matrix = opt_matrix
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = reprojection_error
        self.perspective_transform = perspective_transform
        self.scale = scale

def from_file(filepath):
    """
    Read a calibration object from a file.
    """
    return pickle.load(filepath) 

def save(calib_object, filepath):
    """
    Pickle and store a calibration object.
    """
    pickle.dump(calib_object, filepath)