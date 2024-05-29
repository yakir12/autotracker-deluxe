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
                 scale=None,
                 bbox_width=None,
                 bbox_height=None,
                 chessboard_size=None,
                 chessboard_square_size=None,
                 metadata="",
                 corrective_transform=None,
                 uncorrected_homography=None,
                 adjustment=None):
        """
        :param matrix: The camera matrix
        :param distortion: The distortion coefficients
        :param opt_matrix: The 'new optimal' camera matrix
        :param rvecs: Rotation vectors
        :param tvecs: Translation vectors
        :param reprojection_error: Reprojection error returned by calibrateCamera
        :param perspective_transform: The extrinsic camera perspective transformation (3x3 matrix)
        :param scale: The scale parameter, px / scale = mm
        :param width: The width of the bounding box which contains all points in the
                      transformed frame. Required if you ever want to show a calibrated
                      frame.
        :param height: The height of the bounding box which contains all points in
                       the transformed frame. Required if you ever want to show a
                       calibrated frame.
        :param chessboard_size: The chessboard size (inner corners)
        :param chessboard_square_size: The chessboard square size in mm
        :param metadata="": Textual information describing this calibration
        :param corrective_transform=None: Additional perspective transformation to recentre the calibration board.
        :param uncorrected_homography: The original, unmodified perspective transform
        :param adjustment: Additive adjustment to be applied once the perspective transformation has occurred.
        """

        self.camera_matrix = matrix
        self.distortion = distortion
        self.opt_matrix = opt_matrix
        self.rvecs = rvecs
        self.tvecs = tvecs
        self.reprojection_error = reprojection_error
        self.perspective_transform = perspective_transform
        self.scale = scale
        self.bbox_width = bbox_width
        self.bbox_height = bbox_height
        self.chessboard_size = chessboard_size
        self.chessboard_square_size = chessboard_square_size
        self.corrective_transform=corrective_transform,
        self.uncorrected_homography=uncorrected_homography
        self.adjustment=adjustment

        # Information about the calibration which should be set by the user
        # on generation.
        self.__metadata = metadata

def from_file(filepath):
    """
    Read a calibration object from a file.
    """
    with open(filepath, 'rb') as f:
        calib = pickle.load(f) 
    return calib

def save(calib_object, filepath):
    """
    Pickle and store a calibration object.
    """
    with open(filepath, 'wb') as f:
        pickle.dump(calib_object, f)

def verify_calibration(filepath):
    """
    Check that the file at a given path was actually a calibration file.
    """
    calib = from_file(filepath)
    return isinstance(calib, Calibration)