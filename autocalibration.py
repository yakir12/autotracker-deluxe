"""
Test script for honing calibration procedure.

Randomly extract n frames from a test calibration video which have the chessboard
present. An extrinsic frame is hard coded for the test video

"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def define_object_chessboard(n_rows, n_columns, square_size):
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

def generate_calibration_from_cache(object_chessboard,
                                    chessboard_size,
                                    square_size,
                                    cache_path='calibration_image_cache'):
    # Work out object points
    # Flip chessboard
    # object_chessboard = ((np.ones(object_chessboard.shape) * 255) - object_chessboard).astype(np.uint8)
    object_points = []
    object_success, obj_points = cv2.findChessboardCorners(object_chessboard, 
                                                           chessboard_size)

    # OpenCV wants the object points as an array of the form 
    # [[x1, y1, 0], ... , [xN, yN, 0]] (which isn't what the above function returns)
    # This list manipulation augments each point with a 0 and strips out one of
    # the additional dimensions given by findChessboardCorners.
    obj_points =\
          np.array([ (np.append(op[0], 0.0)) for op in obj_points]).astype(np.float32)

    #
    # Intrinsic calibration
    #

    # Read in pattern corners from cache
    corner_file_list = os.listdir(os.path.join(cache_path, 'intrinsic', 'corners'))
    
    image_points = []
    for file in corner_file_list:
        filepath = os.path.join(cache_path, "intrinsic", "corners", file)
        points = np.load(filepath, allow_pickle=True)
        image_points.append(points)

        # Note, OpenCV requires the number of sets of object points and the 
        # number of sets of image points to be the same. I don't know why
        # as object points are presumably constant and never change. To
        # satisfy OpenCV we replicate the object points for each set of 
        # image points.
        object_points.append(obj_points)

    # Determine image size
    imagepath = os.path.join(cache_path, 'intrinsic', '000.png')
    sample_frame = cv2.imread(imagepath)
    frame_size = sample_frame.shape[:2]

    # Compute camera calibration.

    # The camera model assumes more variability than should actually be
    # possible with a standard consumer video camera which is correctly 
    # configured for experiments.

    # Avoid changing k2 and k3 radial distortion parameters and assume no
    # tangent distortion. This is drawn from Yakir and corroberated with some
    # online discussion on OpenCV calibration
    flags = cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K2 + cv2.CALIB_ZERO_TANGENT_DIST
    
    rproj_err, mtx, dist, rvecs, tvecs =\
          cv2.calibrateCamera(object_points, 
                              image_points,
                              frame_size,
                              None,
                              None,
                              flags=flags)
    optmtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, frame_size, 1, frame_size)

    #
    # Extrinsic calibration (camera perspective transformation)
    # 
    extrinsic_path = os.path.join(cache_path, 'extrinsic', '000.png')
    extrinsic_img = cv2.imread(extrinsic_path)
    undistorted_extrinsic =\
          cv2.undistort(extrinsic_img, mtx, dist, newCameraMatrix=optmtx)
    
    success, ext_points = cv2.findChessboardCorners(undistorted_extrinsic,
                                                    chessboard_size)
    
    if not success:
        print("Extrinsic calibration failed: chessboard could not be found" +
              " in undistorted image. Either the intrinsic calibration is bad" +
              " and has distorted the chessboard, or there is no chessboard " +
              "present.")
        return None
    
    # Compute the perspective transformation between an undistorted image plane and the 
    # ground plane.
    
    homography, _ = cv2.findHomography(ext_points, obj_points)  
   
    dsize = (undistorted_extrinsic.shape[1], undistorted_extrinsic.shape[0])

    # Adjust transformation to keep the full frame in the destination image dimensions. 
    # This is Vishaal's code with different variable names.

    # Place target image corners (as homogenous coordinates) into a matrix
    image_dimensions =\
          np.transpose(np.array([[0, 0, 1],
                       [undistorted_extrinsic.shape[1], 0, 1],
                       [undistorted_extrinsic.shape[1], undistorted_extrinsic.shape[0], 1],
                       [0, undistorted_extrinsic.shape[0], 1]]))
    
    # Remap them using the homography 
    transformed_image_dimensions = np.matmul(homography,image_dimensions)

    # Reformat point representation for OpenCV
    image_dimensions = np.transpose(image_dimensions[0:2])
    image_dimensions = np.float32(image_dimensions)       
    transformed_image_dimensions = np.transpose(transformed_image_dimensions[0:2])
    transformed_image_dimensions = np.float32(transformed_image_dimensions)

    # Work out the transformation between the remapped bounds and the original bounds.
    corrective_transform =\
          cv2.getPerspectiveTransform(transformed_image_dimensions, image_dimensions)

    # Apply the corrective transform to the original homography
    homography = np.matmul(corrective_transform,homography)

    #
    # Scale - scale transformation between undistorted perspective shifted (calibrated)
    # image and the object checkerboard.
    #
    calibrated_extrinsic_frame = cv2.warpPerspective(undistorted_extrinsic,
                                                     homography,
                                                     dsize)
    
    success, img_scale_points = cv2.findChessboardCorners(calibrated_extrinsic_frame, 
                                                          chessboard_size)
    
    #
    # Estimate the average square size detected in the image
    #
    # Compute raw differences between pair of corner coordinates detected in the first row
    # of the chessboard. 
    img_scale_points = np.squeeze(img_scale_points)
    raw_differences =\
        img_scale_points[1:chessboard_size[0]-1] - img_scale_points[0:chessboard_size[0] - 2]

    # Compute distances between corners from raw differences (square, sum, square root)
    distances = np.linalg.norm(raw_differences, axis=1)

    # Mean calibration square edge length in pixels in the calibrated image
    mean_distance = np.mean(distances)

    # Determine scaling parameter
    scale = mean_distance / square_size

    estimated_edge_length = np.linalg.norm(img_scale_points[chessboard_size[0]-1] - img_scale_points[0])/scale
    true_edge_length = square_size * (chessboard_size[0] - 1)

    # scale = px/mm -> x px / scale = y mm approximate true distance.
    print("= Calibration check! =")
    print("Your calibration board is {} columns by {} rows".format(chessboard_size[0], chessboard_size[1]))
    print("Your square size is {}mm".format(square_size))
    print("Top edge is {} squares".format(chessboard_size[0] - 1))
    print("Length of top edge in mm (true : estimated) -> ({} : {})".format(true_edge_length, estimated_edge_length))
    

    #input()

    #
    ## TROUBLESHOOTING IMAGE DISPLAY ##
    #
    sample_image = cv2.imread('calibration_image_cache/extrinsic/000.png')
    imheight = sample_image.shape[0]
    
    border = (255 * np.ones((imheight, 100, 3))).astype(np.uint8) # generate white border
    
    calibrated_extrinsic_frame = cv2.drawChessboardCorners(calibrated_extrinsic_frame,
                                                           chessboard_size,
                                                           img_scale_points,
                                                           success)
    complete_frame = np.concatenate((sample_image, 
                                     border, 
                                     undistorted_extrinsic, 
                                     border, 
                                     calibrated_extrinsic_frame),  axis=1)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
        cv2.imshow('frame', complete_frame)
        if cv2.waitKey(1) == 'q':
            break
    

    # Compile results
    results = dict()
    results["mtx"] = mtx
    results["dist"] = dist
    results["optmtx"] = optmtx
    results["rvecs"] = rvecs
    results["tvecs"] = tvecs
    results["rproj_err"] = rproj_err
    results["homography"] = homography
    results["scale"] = scale

    return results

def check_calibration(example_image_path, calibration):
    """
    Load an example image, and show the outcome of a given calibration on 
    that image.

    :param example_image_path: A filepath to an extrinsic calibration image which
                               can be used to examine distortion.
    :param calibration: A Calibration object which can be used to provide arguments
                        for cv2.undistort and cv2.warpPerspective. 
    """
    chessboard_size = calibration.chessboard_size
    square_size = calibration.square_size
    sample_image = cv2.imread(example_image_path)
    dsize = sample_image.shape
    imheight = dsize[0]
    
    undistorted_extrinsic =\
          cv2.undistort(sample_image, 
                        calibration.camera_matrix, 
                        calibration.distortion, 
                        newCameraMatrix=calibration.opt_matrix) 

    calibrated_extrinsic_frame =\
          cv2.warpPerspective(undistorted_extrinsic,
                              calibration.perspective_transform,
                              dsize)
    
    success, img_scale_points = cv2.findChessboardCorners(calibrated_extrinsic_frame,
                                                          chessboard_size)
    
    calibrated_extrinsic_frame = cv2.drawChessboardCorners(calibrated_extrinsic_frame,
                                                           chessboard_size,
                                                           img_scale_points,
                                                           success)
    
    border = (255 * np.ones((imheight, 100, 3))).astype(np.uint8) # generate white border    
    complete_frame = np.concatenate((sample_image, 
                                     border, 
                                     undistorted_extrinsic, 
                                     border, 
                                     calibrated_extrinsic_frame),  axis=1)
    
    cv2.putText(complete_frame, 
                'See result in terminal.',
                (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (255,255,255), 
                2, 
                cv2.LINE_AA)

    estimated_edge_length = np.linalg.norm(img_scale_points[chessboard_size[0]-1] - img_scale_points[0])/scale
    true_edge_length = square_size * (chessboard_size[0] - 1)

    # scale = px/mm -> x px / scale = y mm approximate true distance.
    print("")
    print("= Calibration check! =")
    print("Your calibration board is {} columns by {} rows".format(chessboard_size[0], chessboard_size[1]))
    print("Your square size is {}mm".format(square_size))
    print("Top edge is {} squares".format(chessboard_size[0] - 1))
    print("Length of top edge in mm (true : estimated) -> ({} : {})".format(true_edge_length, estimated_edge_length))    
    print("")
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    
    
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
        cv2.imshow('frame', complete_frame)
        if cv2.waitKey(1) == 'q':
            break
        
    pass

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

    refresh_cache=False
    if refresh_cache:
        cache_calibration_video_frames(video, 
                                       object_chessboard, 
                                       chessboard_size, 
                                       N=30)
    
    calibration = generate_calibration_from_cache(object_chessboard,
                                                  chessboard_size,
                                                  square_size)
    

    # sample_image = cv2.imread('calibration_image_cache/intrinsic/000.png')
    # imheight = sample_image.shape[0]
    
    # border = (255 * np.ones((imheight, 100, 3))).astype(np.uint8) # generate white border
    
    # print(sample_image.shape)

    # dst = cv2.undistort(sample_image, 
    #                     calibration['mtx'],
    #                     calibration['dist'],
    #                     None, 
    #                     calibration['optmtx'])
    
    # print(dst.shape)

    # complete_frame = np.concatenate((sample_image, border, dst),  axis=1)

    # print(calibration["rvecs"])

    # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    # while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
    #     cv2.imshow('frame',complete_frame)
    #     if cv2.waitKey(1) == 'q':
    #         break
    

    

                                                
    
    


    
