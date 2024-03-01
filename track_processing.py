import numpy as np
import pandas as pd
import cv2
from scipy.interpolate import interp1d


def calibrate_tracks(camera_matrix, 
                     distortion_coefficients, 
                     raw_track_file, 
                     dest_filepath,
                     homography=None):
    """
    Calibrate raw tracks from autotracking to store them in world coordinates.
    The raw track file is assumed to be a csv.

    :param camera_matrix: The camera calibration matrix
    :param distance_coefficients: The distance coefficients computed during 
                                  calibration.
    :param raw_track_file: The raw track in pixelcoordinates (csv)
    :param dest_filepath: The output file for the calibrated tracks
    :param homography: The transformation matrix required to project into 
                       world coordinates. If not supplied, the conversion will
                       not be performed.
    """
    raw_data = pd.read_csv(raw_track_file, index_col=[0])
    calibrated_data = pd.DataFrame(columns=raw_data.columns, 
                                   index=raw_data.index)
    columns = list(raw_data.columns)

    H = homography # Rename for ease

    # Iterate over raw data and calibrate each set of x,y points
    col_idx = 0
    while col_idx < len(columns):
        # Extract datapoints and convert to numpy arrays
        x_data = raw_data.loc[:,columns[col_idx]]
        y_data = raw_data.loc[:,columns[col_idx+1]]
        x_data = x_data.to_numpy(dtype=np.float64, na_value=np.nan)
        y_data = y_data.to_numpy(dtype=np.float64, na_value=np.nan)

        # Check lengths match (fail otherwise)
        assert len(x_data) == len(y_data)

        # Pack into 2xN array for calibration
        points = np.stack((x_data, y_data))

        print(points)
        # Calibrate points, returns Nx2
        calibrated_points =\
              cv2.undistortImagePoints(points, 
                                       cameraMatrix=camera_matrix,
                                       distCoeffs=distortion_coefficients)
        
        # Convert to 2xN
        calibrated_points = calibrated_points.T

        # If we have the homography to translate into world coordinates.
        if not (H == None):
            # Generate sequence of 1s to be added to each point
            ones = np.ones(points.shape[1])

            # Each coordinate now [x,y,1], full structure is Nx3
            calibrated_points = np.stack((calibrated_points, ones)).T
            
            # Map the homography transformation onto every point
            # Should return a 3xN matrix of the transformed points
            calibrated_points =\
                np.array(list(map(lambda x: np.dot(H,x), calibrated_points)))

        # Insert calibrated x and y values into new dataframe.
        calibrated_data.loc[:, columns[col_idx]] = calibrated_points[0]
        calibrated_data.loc[:, columns[col_idx+1]] = calibrated_points[1]

        col_idx += 2 # Iterate over pairs of columns

    # Write out new dataframe.
    calibrated_data.to_csv(dest_filepath)

def smooth_tracks(track_file, dest_file=None):
    """
    Apply smoothing to tracks.

    If dest_file == None then the destination filename will be based on the
    track_file.    

    :param track_file: The csv track file you wish to use
    :param dest_file: A destination file
    """
    pass

def zero_tracks(raw_track_file, dest_filepath, origin=(0,0)):
    """
    Normalise all tracks in a file such that they start from origin. The first
    point in a track is assumed to be the origin.

    :param track_file: The csv track file you wish to use
    :param dest_file: A destination file
    :param origin: The desired origin function point
    """
    data = pd.read_csv(raw_track_file, index_col=[0])
    zeroed_data = pd.DataFrame(columns=data.columns, index=data.index)
    columns = list(data.columns)

    # Iterate over raw data and calibrate each set of x,y points
    col_idx = 0
    while col_idx < len(columns):
        # Extract datapoints and convert to numpy arrays
        x_data = data.loc[:,columns[col_idx]]
        y_data = data.loc[:,columns[col_idx+1]]
        x_data = x_data.to_numpy(dtype=np.float64, na_value=np.nan)
        y_data = y_data.to_numpy(dtype=np.float64, na_value=np.nan)

        # Check lengths match (fail otherwise)
        assert len(x_data) == len(y_data)

        # Determine X and Y offset from desired origin
        x_offset = x_data[0] - origin[0]
        y_offset = y_data[0] - origin[1]
        
        # Apply translate all points
        norm_x = x_data - x_offset
        norm_y = y_data - y_offset

        # Insert calibrated x and y values into new dataframe.
        zeroed_data.loc[:, columns[col_idx]] = norm_x
        zeroed_data.loc[:, columns[col_idx+1]] = norm_y

        col_idx += 2 # Iterate over pairs of columns

    # Write out new dataframe.
    zeroed_data.to_csv(dest_filepath)