from tkinter import messagebox

import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import cv2
from scipy.interpolate import UnivariateSpline

import matplotlib.pyplot as plt


from dtrack_params import dtrack_params
from project import project_file

import calibration as calib

def calibrate_tracks(calibration: calib.Calibration, 
                     raw_track_filepath: str, 
                     dest_filepath: str):
    """
    Calibrate raw tracks from autotracking to store them in world coordinates.
    The raw track file is assumed to be a csv.

    :param calibration: A calibration object (see calibration.py)
    :param raw_track_file: The raw track in pixelcoordinates (csv)
    :param dest_filepath: The output file for the calibrated tracks
    """
    camera_matrix = calibration.camera_matrix
    homography = calibration.perspective_transform
    distortion_coefficients = calibration.distortion
    scale = calibration.scale

    raw_data = pd.read_csv(raw_track_filepath, index_col=[0])
    calibrated_data = pd.DataFrame(columns=raw_data.columns, 
                                   index=raw_data.index)
    columns = list(raw_data.columns)

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

        #
        # Intrinsic calibration (camera distortion)
        #  

        # Pack into 2xN array for calibration
        points = np.stack((x_data, y_data))

        # Calibrate points, returns Nx1x2
        
        calibrated_points =\
              cv2.undistortImagePoints(points, 
                                       cameraMatrix=camera_matrix,
                                       distCoeffs=distortion_coefficients)
        
        # Remove dimension added by OpenCV and transpose to 2XN
        calibrated_points = np.squeeze(calibrated_points)
        calibrated_points = calibrated_points.T

        #
        # Extrinsic calibration (camera perspective)
        # 

        # Generate sequence of 1s to be added to each point
        ones = np.ones((1,calibrated_points.shape[1]))

        # Each coordinate now [x,y,1], full structure is Nx3
        calibrated_points = np.concatenate((calibrated_points, ones)).T
        
        # Map the homography transformation onto every point
        # Should return a 3xN matrix of the transformed points
        
        calibrated_points =\
            np.array(list(map(lambda x: np.dot(homography, x), 
                              calibrated_points)))

        calibrated_points = calibrated_points.T

        # scale = px/mm -> (x px / scale) = y mm
        # These coordinates should be in mm.
        scaled_calibrated_points = calibrated_points / scale

        # Insert calibrated x and y values into new dataframe.
        calibrated_data.loc[:, columns[col_idx]] = scaled_calibrated_points[0]
        calibrated_data.loc[:, columns[col_idx+1]] = scaled_calibrated_points[1]

        col_idx += 2 # Iterate over pairs of columns

    # Write out new dataframe.
    calibrated_data.to_csv(dest_filepath)

def smooth_tracks(track_file, 
                  dest_filepath):
    """
    Apply smoothing to tracks.

    If dest_file == None then the destination filename will be based on the
    track_file.    

    :param track_file: The csv track file you wish to use
    :param dest_file: A destination file
    """
    data = pd.read_csv(track_file, index_col=[0])

    smoothed_index = np.arange(data.index.size)
    smoothed_data = pd.DataFrame(columns=data.columns, index=smoothed_index)

    columns = list(data.columns)    

    # Iterate over raw data and calibrate each set of x,y points
    col_idx = 0
    while col_idx < len(columns):
        # Extract datapoints and convert to numpy arrays
        x_data = data.loc[:,columns[col_idx]]
        y_data = data.loc[:,columns[col_idx+1]]
        x_data = x_data.to_numpy(dtype=np.float64, na_value=np.nan)
        y_data = y_data.to_numpy(dtype=np.float64, na_value=np.nan)

        # Filter out NaN values, but retain count for re-augmentation
        full_len_x = x_data.size

        x_data = x_data[~np.isnan(x_data)]
        y_data = y_data[~np.isnan(y_data)]

        # Check lengths match (fail otherwise)
        assert len(x_data) == len(y_data)

        # Work in arbitrary time units. Can combine with FPS later to get
        # true time.
        duration = x_data.size
        t = np.arange(duration)

        # Create a smoothing spline for the data
        degree = 3
        x_spline = UnivariateSpline(t, x_data, k=degree)
        y_spline = UnivariateSpline(t, y_data, k=degree)

        # Compute spline for given time points.
        x_smooth = x_spline(t)
        y_smooth = y_spline(t)

        # Re-augment tracks with NaN values to preserve length and avoid index
        # problems with pandas.
        padding_size = (full_len_x - x_data.size)
        padding = np.array([np.nan for n in range(padding_size)])

        x_smooth = np.concatenate((x_smooth, padding))
        y_smooth = np.concatenate((y_smooth, padding))

        # Insert x with smoothed y values
        smoothed_data.loc[:, columns[col_idx]] = x_smooth
        smoothed_data.loc[:, columns[col_idx+1]] = y_smooth

        col_idx += 2 # Iterate over pairs of columns    
    
    # Write out new dataframe.
    smoothed_data.to_csv(dest_filepath)        

def zero_tracks(raw_track_file, dest_filepath, origin=(0,0)):
    """
    Normalise all tracks in a file such that they start from origin. The first
    point in a track is assumed to be the origin.

    :param track_file: The csv track file you wish to use
    :param dest_file: A destination file
    :param origin: The desired origin point
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

def analyse_tracks(input_filepath, dest_filepath):
    """
    Compute some basic summary stats on the tracks and store these in a file.
    Currently computing path length, displacement, straightness, time to exit, 
    and speed. These are stored in a dataframe which is output as CSV.
    :param input_filepath: The path to the CSV file you want to use for analysis.
    :param dest_filepath: The path where you want to store the statistics file.
    """
    data = pd.read_csv(input_filepath, index_col=[0])
    columns = list(data.columns)
    
    # Generate index for new dataframe
    n_tracks = int(len(columns)/2)
    track_indices = np.arange(0, n_tracks)
    track_labels = ["Track_" + str(x) for x in track_indices]
    track_labels.append("Mean")
    track_labels.append("Std. Dev.")

    # Generate analysis columns for new dataframe
    headers = ["Length (m)", 
               "Displacement (m)", 
               "Straightness", 
               "Time to exit (s)", 
               "Speed (m/s)"]

    stats = pd.DataFrame(index=track_labels, columns=headers)

    col_idx = 0
    while col_idx < len(columns):
        track_no = columns[col_idx].split("_")[1]
        track_label = "Track_" + track_no
        
        xs = data.loc[:, columns[col_idx]].to_numpy()
        ys = data.loc[:, columns[col_idx+1]].to_numpy()

        assert(len(xs[~np.isnan(xs)])  == len(ys[~np.isnan(ys)]))

        # Strip out NaN values
        xs = xs[~np.isnan(xs)]
        ys = ys[~np.isnan(ys)]

        # Path length
        x_dists = np.power(np.ediff1d(xs), 2)
        y_dists = np.power(np.ediff1d(ys), 2)
        dists = np.sqrt(np.add(x_dists, y_dists))

        # Calibrated tracks are stored in mm
        path_length = np.sum(dists) / 1000 
        stats.loc[track_label, 'Length (m)'] = path_length

        # Displacement
        euc_dist = lambda p, q: np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
        start_point = (xs[0], ys[0])
        end_point = (xs[-1], ys[-1])
        displacement = euc_dist(start_point, end_point) / 1000
        stats.loc[track_label, 'Displacement (m)'] = displacement

        # Straightness
        straightness = displacement/path_length
        stats.loc[track_label, 'Straightness'] = straightness

        if project_file["track_fps"] != -1:
            # Total time to complete this track assuming one track point
            # per frame
            time_for_track = len(xs) / project_file["track_fps"] 
            stats.loc[track_label, 'Time to exit (s)'] = time_for_track

            # Speed in m/s
            speed = path_length / time_for_track
            stats.loc[track_label, 'Speed (m/s)'] = speed

        col_idx += 2
    
    # Compute means and standard deviations
    for col in stats.columns:
        stats.loc['Mean', col] = np.mean(stats.loc[:, col])
        stats.loc['Std. Dev.', col] = np.std(stats.loc[:, col])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
        print(stats)

    stats.to_csv(dest_filepath)

    print("Summary statistics stored in: {}".format(dest_filepath))
    
    

def plot_tracks(input_file, 
                draw_arena=False, 
                arena_radius=50, 
                draw_mean_displacement=False,
                scale=1000):
    """
    Helper method to test calibration, this is only intended to check distance
    tranformations have been performed successfully, this is not for any 
    formal analysis.

    :param input_file: The CSV file you want to use as the underlying data.
    :param draw_arena: Draw a circle on the plot of the same radius as the
                       arena used in the experiment.
    :param arena_radius: The radius of the arena in cm.
    """

    data = pd.read_csv(input_file, index_col=[0])
    columns = list(data.columns)

    # Convert to metres
    arena_radius = arena_radius * 10
    
    mosaic = [['si_tracks']]
    fig, axs = plt.subplot_mosaic(mosaic)
    ax = axs['si_tracks']

    if draw_arena:
        def define_circle(radius):
            ths = np.linspace(0, 2*np.pi, 100)     
            rs = radius * np.ones(100)
            xs = [r*np.cos(th) for (th, r) in zip(ths, rs)]
            ys = [r*np.sin(th) for (th, r) in zip(ths, rs)]
            return xs, ys

        arena = define_circle(arena_radius)
        starter = define_circle(50) # 5cm starting circle 
        ax.plot(arena[0], arena[1], color='k')
        ax.plot(starter[0], starter[1], color='k')
        ax.set_aspect('equal') # If we're drawing the arena, fair assumption

    if dtrack_params["options.processing.plot_grid"]:
        ax.grid()
        
    displacements = []
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
        x_data /= scale
        y_data /= scale

        roll_no = columns[col_idx].split("_")[1]
        ax.plot(x_data, y_data, alpha=0.5, label="Track {}".format(roll_no))
        
        # Track average displacement

        # Find last non NaN element 
        x_end = x_data[~np.isnan(x_data)][-1]        
        y_end = y_data[~np.isnan(y_data)][-1]        
        x_disp = (x_data[0] - x_end)**2
        y_disp = (y_data[0] - y_end)**2

        # Compute and store displacement
        disp = np.sqrt(x_disp + y_disp)
        displacements.append(disp)

        col_idx += 2 # Iterate over pairs of columns

    if dtrack_params["options.processing.include_legend"]:
        ax.legend()

    print("Plotted: {}".format(input_file))

    filename = dtrack_params["options.processing.filename"] + "." +\
               dtrack_params["options.processing.filetype"]
    filepath = os.path.join(dtrack_params["project_directory"], filename)

    plt.savefig(filepath, dpi=400, bbox_inches="tight")

    print("Plot saved as {}".format(filepath))

    plt.show()


def calibrate_and_smooth_tracks():
    calibration_filepath = project_file["calibration_file"]
    
    # Check for calibration file
    if not os.path.exists(calibration_filepath):
        msg = "This project has no calibration file, use the calibration manager" +\
              " to generate or import one."
        messagebox.showerror(title="No calibration file!",
                             message=msg)
        return

    # Load calibration file
    calibration = calib.from_file(calibration_filepath)
    
    raw_data_filepath = os.path.join(dtrack_params["project_directory"],
                                     'raw_tracks.csv')
    calibrated_filepath = os.path.join(dtrack_params["project_directory"],
                                       'calibrated_tracks.csv')
    smoothed_filepath = os.path.join(dtrack_params["project_directory"],
                                     'smoothed_tracks.csv')
    stats_filepath = os.path.join(dtrack_params["project_directory"],
                                  "summary_statistics.csv")

    calibrate_tracks(calibration,
                     raw_data_filepath,
                     calibrated_filepath)
    
    zeroed_filepath = calibrated_filepath
    if dtrack_params['options.processing.zero']:
        zeroed_filepath = os.path.join(dtrack_params["project_directory"],
                                       'zeroed_tracks.csv')    
        zero_tracks(calibrated_filepath, zeroed_filepath)
    
    smooth_tracks(zeroed_filepath, smoothed_filepath)

    analyse_tracks(smoothed_filepath, stats_filepath)

    plot_tracks(smoothed_filepath)