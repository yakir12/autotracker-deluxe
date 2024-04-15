"""
Test script for honing calibration procedure.

Randomly extract n frames from a test calibration video which have the chessboard
present. An extrinsic frame is hard coded for the test video

"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def define_object_chessboard(n_rows, n_columns, square_size,):
    """
    Create 'object' chessboard image such that one pixel == one millimetre. 
    This can be used to find the homography between a calibrated image and
    the world (object) space such that we can transform pixels into mm for
    analysis.

    :param n_rows: The number of rows in the pattern.
    :param n_columns: The number of columns in the pattern.
    :param square_size: The size of each chessboard square in mm.
    :return: The chessboard image and a tuple storing the chessboard size (-1 in each dim)    
    """
    # Create binarised chessboard
    rows_grid, columns_grid = np.meshgrid(range(n_rows), range(n_columns), indexing='ij')
    high_res_chessboard = np.mod(rows_grid, 2) + np.mod(columns_grid, 2) == 1

    # Create block matrix at full resolution (1px/mm).
    square = np.ones((square_size,square_size))
    chessboard = np.kron(high_res_chessboard, square)

    # Number of *inner* corners per dimension.
    chessboard_size = (n_columns-1, n_rows-1)

    return chessboard, chessboard_size


def undistort_image(frame, mtx=None, dist=None):
    # Include camera matrix and distortion
    frame_dist = cv2.undistort(frame, 
                               cameraMatrix=mtx,
                               distCoeffs=dist)
    return frame_dist

def cache_calibration_video_frames(video_path, 
                                   object_chessboard,
                                   chessboard_size,
                                   N=15, 
                                   extrinsic_frames=0,
                                   frame_cache='calibration_image_cache'):
    """
    Select N frames from a calibration video where the chessboard is successfully
    found.

    :param video_path: The filepath to the calibration video
    :param object_chessboard: An OpenCV image of the object-space chessboard.
    :param chessboard_size: The dimensions of the chessboard (inner corners).
    :param N: the number of frames you want to find.
    :param extrinsic_frame: The index of a frame which can be used for extrinsic
                            calibration (the chessboard is on the ground). This
                            is guessed to be the first frame but SHOULD BE 
                            PROVIDED BY THE CALLER!
    :param frame_cache: The caching directory to use for calibration frames.
    :returns: The filepath to the frames.
    """

    # Ensures randomness is repeatable.
    random_state = np.random.RandomState()


    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    sample_indices = random_state.choice(range(int(frame_count)),
                                         size=int(N),
                                         replace=False)


    # Cache appropriate images for intrinsic calibration
    failed_indices = []
    idx = 0
    while idx < len(sample_indices):
        frame_idx = sample_indices[idx]

        print("Checking frame {}".format(frame_idx))

        # Set capture position and get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()

        chessboard_found, corners = cv2.findChessboardCorners(frame,
                                                              chessboard_size)
        if not chessboard_found:
            print("Failed to find chessboard in frame, trying a new frame...")
            failed_indices.append(frame_idx) # Keep track of failed frames
            new_choice = frame_idx
            
            # Look for a new (random) frame index which hasn't been tried already
            # and isn't in the current sample_indices list.
            while ((new_choice in failed_indices) or (new_choice in sample_indices)):
                new_choice = random_state.choice(range(int(frame_count)), size=1)

            # Replace the current frame with the new one to try again.
            sample_indices[idx] = new_choice
            
            continue # continue without index increment to try again.

        # If the chessboard was found, then save a copy of the image with
        # the corners drawn and a clean copy.
        filepath = os.path.join(frame_cache, "intrinsic", "{:03d}.png".format(idx))
        corner_file = os.path.join(frame_cache, "intrinsic", "corners", "{:03d}.dat".format(idx))
        print("Chessboard found, caching image at {}".format(filepath))
        cv2.imwrite(filepath, frame)
        corners.dump(corner_file)
        idx += 1

    # Cache extrinsic image, these are pre-selected so if the chessboard can't
    # be found, inform the user and move on. Only issue is if there was no
    # successful extrinsic frame.
    extrinsic_frames = np.atleast_1d(extrinsic_frames) # Make sure this is in array format
    ext_success = False # Overall extrinsic calibration success
    count = 0
    for frame_idx in extrinsic_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        print("Extrinsic, checking frame {}".format(frame_idx))
        _, frame = cap.read()
        
        chessboard_found, corners = cv2.findChessboardCorners(frame,
                                                              chessboard_size)
        ext_success |= chessboard_found

        if not chessboard_found:
            print("Chessboard not found in frame {} for extrinsic calibration.")
            continue

        filepath = os.path.join(frame_cache, "extrinsic", "{:03d}.png".format(count))
        corner_file = os.path.join(frame_cache, "extrinsic", "corners", "{:03d}.dat".format(count))
        print("Chessboard found, caching image at {}".format(filepath))
        cv2.imwrite(filepath, frame)
        corners.dump(corner_file)

        count += 1
        
    if not ext_success:
        print("Warning: Chessboard not found in any frames chosen for extrinsic calibration!")

if __name__ == "__main__":
    # Use calibration info to work out homography
    # Detect checkerboard in image then report distance between corners.

    # Selected test video
    video = "/home/robert/postdoc/source/output.mov"
    
    # Chessboard parameters
    n_rows = 6
    n_cols = 9
    square_size = 39

    # Define an object chessboard image and convert to OpenCV-compatible type.
    object_chessboard, chessboard_size =\
          define_object_chessboard(n_rows, n_cols, square_size)
    object_chessboard = object_chessboard.astype(np.uint8) * 255

    refresh_cache=True
    if refresh_cache:
        cache_calibration_video_frames(video, 
                                       object_chessboard, 
                                       chessboard_size, 
                                       N=30)
