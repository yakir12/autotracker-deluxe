"""
autocalibration.py

Back-end for all functions relating to camera calibration.

Related:
- autocalibration_tool.py -> Tkinter front-end to access this functionality.
- calibration.py -> Class to hold Calibration data.
- old_calibration.py -> Old calibration procedure (selecting frames live from OpenCV).
"""

import numpy as np
import cv2
import os
import shutil
import calibration
import textwrap

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

def select_extrinsic_frame(video_path, calibration_cache, chessboard_size):    
    """
    Spawn an OpenCV window to allow the user to select an extrinsic calibration
    image from their calibration video.

    :param video_path: The file path of the calibration video.
    :param calibration_cache: The cache location where the frame should be stored.
    """
    window = "Select extrinsic calibration frame"
    trackbar = 'trackbar'
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    
    def trackbar_callback(trackbar_value):
        """
        Trackbar callback function (because python doesn't allow
        multiline lambdas).

        :param trackbar_value: The position of the trackbar
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
        frame = cap.read()[1]
        cv2.putText(frame, 
                    'Select a frame where the calibration board is on the ground',
                    (50,50), # Origin
                    cv2.FONT_HERSHEY_SIMPLEX, # Font 
                    1, # Font scale
                    (255,255,255), # colour
                    2, # Line thickness
                    cv2.LINE_AA # Line type
                    )
        cv2.putText(frame, 
                    'Press s to save the current frame and quit, q to quit without saving',
                    (50,100), # Origin
                    cv2.FONT_HERSHEY_SIMPLEX, # Font 
                    1, # Font scale
                    (255,255,255), # colour
                    2, # Line thickness
                    cv2.LINE_AA # Line type
                    )
                
        chessboard_found, corners = cv2.findChessboardCorners(frame,
                                                              chessboard_size)               
        cv2.drawChessboardCorners(frame, 
                                  chessboard_size, 
                                  corners, 
                                  patternWasFound=chessboard_found)
        
        
        cv2.imshow(window, frame)

    cv2.createTrackbar(trackbar, 
                       window, 
                       1, 
                       length,
                       trackbar_callback)

    frame_was_set = False
    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE):
        kp = cv2.waitKey(1)
        if kp == ord('s'):
            # Re-read frame to store without text
            cap.set(cv2.CAP_PROP_POS_FRAMES, 
                    int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
            frame = cap.read()[1]

            # Check for existance of extrinsic calibration directory
            path = os.path.join(calibration_cache, 'extrinsic')
            if not os.path.exists(path):
                # Create extrinsic calibration cache and corner directory
                os.mkdir(path)
                os.mkdir(os.path.join(path, 'corners'))

            # Write image
            cv2.imwrite(os.path.join(path, '000.png'), frame)                

            # Detect chessboard corners and save to file.
            _, corners = cv2.findChessboardCorners(frame,
                                                   chessboard_size)
            corners.dump(os.path.join(path, 'corners', '000.dat'))

            frame_was_set = True
            break
        elif kp == ord('q'):
            # Just break the control loop (close the window without doing anything)
            break
        
    cv2.destroyAllWindows()

    # Return flag and frame idx (each read increments the capture position, so
    # sub 1)
    return frame_was_set, int(cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

def store_corners_from_image_file(image_path,
                                  chessboard_size,
                                  output_directory):
    """
    Given an image filepath, use OpenCV to detect the corners of a chessboard,
    display this for user verification and then store the corners in the 
    output directory if the user is happy with the selection

    :param image_path: The path to the image
    :param chessboard_size: The dimensions of the chessboard we're looking for.
    :param output_directory: The output directory for the corner file (excluding the final filename).
    :return: True if corners were found and user opted to save, False otherwise.
    """

    # Load frame and attempt to find chessboard corners
    frame = cv2.imread(image_path)
    success, corners = cv2.findChessboardCorners(frame, chessboard_size)
    cv2.drawChessboardCorners(frame, chessboard_size, corners, patternWasFound=success)

    if not success:
        # If corners were not found inform the user
        cv2.putText(frame, 
                'No chessboard with dimensions {} by {} found in this frame.',
                (50,50), # Origin
                cv2.FONT_HERSHEY_SIMPLEX, # Font 
                1, # Font scale
                (255,255,255), # colour
                2, # Line thickness
                cv2.LINE_AA # Line type
                )
    else:
        # Otherwise tell the user how to save their selection
        cv2.putText(frame, 
                    'Press s to confirm this is the frame you want to use.',
                    (50,50), # Origin
                    cv2.FONT_HERSHEY_SIMPLEX, # Font 
                    1, # Font scale
                    (255,255,255), # colour
                    2, # Line thickness
                    cv2.LINE_AA # Line type
                    )
    cv2.putText(frame, 
                'Press q to quit without saving',
                (50,100), # Origin
                cv2.FONT_HERSHEY_SIMPLEX, # Font 
                1, # Font scale
                (255,255,255), # colour
                2, # Line thickness
                cv2.LINE_AA # Line type
                )
    
    # Display the frame with a simple control loop
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    store = True
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
        kp = cv2.waitKey(1)

        # If the corners were found and the user selects save then check the
        # required directory exists and save the corners.
        if success and kp == ord('s'):
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            path = os.path.join(output_directory, '000.dat')
            corners.dump(path)
            break
        elif kp == ord('q'):
            # If they quit, set a flag
            store = False
            break

    cv2.destroyAllWindows()


    # Method should return True only if the user selected save and the corners
    # were found. In all other cases, should return False.
    return success and store

               

def cache_calibration_video_frames(video_path, 
                                   chessboard_size,
                                   N=15, 
                                   frame_cache='calibration_image_cache'):
    """
    Select N frames from a calibration video where the chessboard is successfully
    found.

    :param video_path: The filepath to the calibration video
    :param chessboard_size: The dimensions of the chessboard (inner corners).
    :param N: the number of frames you want to find.
    :param frame_cache: The caching directory to use for calibration frames.

    :return: The filepath to the frames.
    """
    
    print("Attempting to build calibration cache")
    print("")

    # Init random generator, can be used to make repeatable randomness for testing
    # by providing a seed.
    random_state = np.random.RandomState()

    # Open OpenCV video capture
    cap = cv2.VideoCapture(video_path)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if N > frame_count:
        print("Cache construction failed!")
        print("You selected N to be greater than the number of frames in the calibration video ({})."
              .format(frame_count))
        print("Try again with reduced N (< {}).".forma(frame_count))

    if not os.path.exists(os.path.join(frame_cache, 'intrinsic')):
        # Create the intrinsic/corners subdirectory.
        os.makedirs(os.path.join(frame_cache, 'intrinsic', 'corners'))
    else:
        # Clear old calibration cache
        shutil.rmtree(os.path.join(frame_cache, 'intrinsic'))
        os.makedirs(os.path.join(frame_cache, 'intrinsic', 'corners'))

    # Compute random frame selection
    sample_indices = random_state.choice(range(int(frame_count)),
                                         size=int(N),
                                         replace=False)

    # Cache appropriate images for intrinsic calibration
    failed_indices = []
    idx = 0
    while idx < len(sample_indices):
        frame_idx = sample_indices[idx]

        print("Chessboard check, frame {}".format(frame_idx))

        # Set capture position and get frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, frame = cap.read()

        chessboard_found, corners = cv2.findChessboardCorners(frame,
                                                              chessboard_size)

        if not chessboard_found:
            print("Failed to find chessboard in frame, trying a new frame...")


            # Note, this may be prone to failure if the chessboard cannot be found
            # in enough frames (software will freeze)
            failed_indices.append(frame_idx) # Keep track of failed frames
            new_choice = frame_idx

            if (len(failed_indices) + len(sample_indices)) > frame_count:
                print("Cache construction failed!")
                print("You asked for {} calibration frames but I could only find {}"
                      .format(N, idx))
                print("Reduce your chosen N to be less than {}".format(idx))
                print("")
                return False
            
            # Look for a new (random) frame index which hasn't been tried already
            # and isn't in the current sample_indices list.
            while ((new_choice in failed_indices) or (new_choice in sample_indices)):
                new_choice = random_state.choice(range(int(frame_count)), size=1)

            # Replace the current frame with the new one to try again.
            sample_indices[idx] = new_choice
            
            continue # continue without index increment to try again.

        # If the chessboard was found, then save the image and the corners to the cache
        filepath = os.path.join(frame_cache, "intrinsic", "{:03d}.png".format(idx))
        corner_file = os.path.join(frame_cache, "intrinsic", "corners", "{:03d}.dat".format(idx))
        print("Chessboard found, caching image at {}".format(filepath))
        cv2.imwrite(filepath, frame)
        corners.dump(corner_file)
        idx += 1

        print("")
        print("Calibration cache constructed succcessfully at:")
        print("{}".format(frame_cache))
        print("")

    return True

def generate_calibration_from_cache(chessboard_columns,
                                    chessboard_rows,
                                    square_size,
                                    cache_path='calibration_image_cache',
                                    metadata=""):
    """
    Generate a calibration file from an image cache (constructed using
    cache_calibration_video_frames()).

    :param chessboard_columns: The number of columns in the chessboard
    :param chessboard_rows: The number of rows in the chessboard
    :param square_size: The square size of the chessboard
    :param cache_path: The location where the calibration cache is stored
    :param metadata: A short description of this calibration
    """

    object_chessboard, chessboard_size =\
        define_object_chessboard(chessboard_rows, chessboard_columns, square_size)
    
    object_chessboard = object_chessboard.astype(np.uint8)

    # Work out object points
    object_points = []
    _, obj_points = cv2.findChessboardCorners(object_chessboard, 
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

    #
    # Compute camera calibration.
    #

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
        return False
    
    # Compute the perspective transformation between an undistorted image plane and the 
    # ground plane.
    homography, _ = cv2.findHomography(ext_points, obj_points)  
   
    dsize = (undistorted_extrinsic.shape[1], undistorted_extrinsic.shape[0])

    # Work out translation correction, so that full arena can be displayed in 
    # a calibrated frame.

    # # Image corners
    # image_bounds = np.array(
    #     [[0, 0, 1],
    #      [0, undistorted_extrinsic.shape[0], 1],
    #      [undistorted_extrinsic.shape[1], 0, 1],
    #      [undistorted_extrinsic.shape[1], undistorted_extrinsic.shape[0], 1]])

    # # Transform image corners to compute where they end up.
    # transformed_image_corners = np.matmul(homography, image_bounds.T)

    # print("")
    # print(image_bounds.T)
    # print(transformed_image_corners)

    # xs = transformed_image_corners[0, :]
    # ys = transformed_image_corners[1, :]

    # print(ys)

    # # Figure out new max and min x and y coordinates.
    # y_bounds = (min(ys), max(ys))
    # x_bounds = (min(xs), max(xs))
    # new_ydim = y_bounds[1] - y_bounds[0]
    # new_xdim = x_bounds[1] - x_bounds[0]

    # # Define new frame size so that full undistorted image can be displayed
    # dsize = (int(new_xdim), int(new_ydim))

    # # Translate so top corner is back in top corner   
    # y_shift = -y_bounds[0]
    # x_shift = -x_bounds[0]

    # print(y_shift)

 
    image_centre = np.array([undistorted_extrinsic.shape[0]/2, undistorted_extrinsic.shape[1]/2])

    chessboard_corners = np.squeeze(obj_points)
    print(chessboard_corners)
  

    chessboard_centre_x =\
        (chessboard_corners[-1][0] - chessboard_corners[0][0])/2
    chessboard_centre_y =\
        (chessboard_corners[-1][1] - chessboard_corners[0][1])/2
    
    chessboard_centre = np.array([chessboard_centre_y, chessboard_centre_x])

    y_shift, x_shift = image_centre - chessboard_centre

    A = [[1, 0, x_shift],
         [0, 1, y_shift],
         [0, 0, 1]]
    
    corrected_homography = np.matmul(A, homography)

    # warped = cv2.warpPerspective(undistorted_extrinsic, corrected_homography, dsize)

    # while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
    #     cv2.imshow('frame', warped)
    #     if cv2.waitKey(1) == ord('q'):
    #         break

    # cv2.destroyAllWindows()
    # return True
  
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

    calib = calibration.Calibration(
        matrix=mtx,
        distortion=dist,
        opt_matrix=optmtx,
        rvecs=rvecs,
        tvecs=tvecs,
        reprojection_error=rproj_err,
        perspective_transform=corrected_homography,
        scale=scale,
        metadata=metadata,
        chessboard_size=chessboard_size,
        chessboard_square_size=square_size,
        uncorrected_homography=homography,
        corrective_transform=A
    )

    calib_filepath = os.path.join(cache_path, 'calibration.dt2c')
    calibration.save(calib, calib_filepath)

    return True


# These globals are only used to allow the calibration check window to update
# on mouse clicks. They should not be used for anything else.
calib_point_ctr = 0
check_calibration_frame = None
calibration_points = []
def on_mouseclick(event, x, y, flags, param):
    global calib_point_ctr
    global check_calibration_frame

    if len(calibration_points) >= 4:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        calib_point_ctr += 1
        print("Pt #{} at ({},{})".format(calib_point_ctr, x, y))
        calibration_points.append((x,y))

        cv2.circle(check_calibration_frame, (x,y), 5, (0,0,255), -1)

    if len(calibration_points) == 2:
        cv2.line(check_calibration_frame, 
                 calibration_points[0], 
                 calibration_points[1],
                 (0, 0, 255),
                 5)
        
    if len(calibration_points) == 4:
        cv2.line(check_calibration_frame, 
                 calibration_points[2], 
                 calibration_points[3],
                 (0, 0, 255),
                 5)
 


    # Would be good to have a right-click to clear option.



def check_calibration(example_image_path, calibration):
    """
    Load an example image, and show the outcome of a given calibration on 
    that image.

    :param example_image_path: A filepath to an extrinsic calibration image which
                               can be used to examine distortion.
    :param calibration: A Calibration object which can be used to provide arguments
                        for cv2.undistort and cv2.warpPerspective. 
    """
    global check_calibration_frame
    global calibration_points

    calibration_points = []

    # Obtain required parameters for the calibration    
    chessboard_size = calibration.chessboard_size
    square_size = calibration.chessboard_square_size
    scale = calibration.scale

    # Read in sample image
    sample_image = cv2.imread(example_image_path)
    dsize = (sample_image.shape[1], sample_image.shape[0])
    imheight = dsize[1]

    # Undistort the sample image
    undistorted_extrinsic =\
          cv2.undistort(sample_image, 
                        calibration.camera_matrix, 
                        calibration.distortion, 
                        newCameraMatrix=calibration.opt_matrix) 

    # Warp the image to give a top-down view.
    check_calibration_frame =\
          cv2.warpPerspective(undistorted_extrinsic,
                              calibration.perspective_transform,
                              dsize)
    
    # Work out the chessboard corners and draw these on the frame.
    success, img_scale_points = cv2.findChessboardCorners(check_calibration_frame,
                                                          chessboard_size)
    
    check_calibration_frame = cv2.drawChessboardCorners(check_calibration_frame,
                                                           chessboard_size,
                                                           img_scale_points,
                                                           success)

    cv2.putText(check_calibration_frame, 
                'LOOK AT THE TERMINAL!',
                (50,50), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, 
                (255,255,255), 
                2, 
                cv2.LINE_AA)

    estimated_edge_length = np.linalg.norm(img_scale_points[chessboard_size[0]-1] - img_scale_points[0])/scale
    true_edge_length = square_size * (chessboard_size[0] - 1)

    # scale = px/mm -> x px / scale = y mm approximate true distance.
    print("")
    print("######################")
    print("# Calibration check! #")
    print("######################")
    print("")
    print("Your calibration board is {} columns by {} rows".format(chessboard_size[0]-1, chessboard_size[1]-1))
    print("Your square size is {}mm".format(square_size))
    print("Top edge is {} squares".format(chessboard_size[0] - 1))
    print("Length of top edge in mm (true : estimated) -> ({} : {})".format(true_edge_length, estimated_edge_length))    
    print("")
    str = "To check distortion over the whole arena, click on four points on the arena radius." +\
          " These points will be used to define two lines, select points such that" +\
          " these lines are not parallel."
    print(textwrap.fill(str, 60))
    print("")
    print("# IMPORTANT #")
    str = "Some of the arena may be cut off in this frame. This does not affect the" +\
          " calibration process (your tracks will be fine). An alternative image correction" +\
          " is in progress to mitigate this problem."
    print(textwrap.fill(str, 60))
    print("")
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    cv2.setMouseCallback('frame', on_mouseclick)
    
    points_drawn = False
    while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
        cv2.imshow('frame', check_calibration_frame)
        if cv2.waitKey(1) == ord('q'):
            break

        if len(calibration_points) == 4 and points_drawn == False:    
            cast_tuple_to_int = lambda t: tuple(map(lambda x: int(x), t))
            def compute_perp_bisector_in_frame(a,b):
                """
                Given points a and b, define the perpendicular bisector of
                the line from a to b.

                :param a: Point 1
                :param b: Point 2
                :return: A tuple containing the boundary points within
                         the check calibration frame for drawing with OpenCV.
                         Also returns the slope and intercept of the bisecting line.
                """
                m0 = (b[0] - a[0]) / (b[1] - a[1]) # Slope of line ab
                p = ((a[0] + b[0])/2, (a[1] + b[1])/2) # Midpoint of line ab
                m0_perp = -m0 # Slope of line perpendicular to ab
                c0_perp = p[1] - (m0_perp*p[0]) # Intercept

                y_bound = check_calibration_frame.shape[0]
                x_bound = check_calibration_frame.shape[1]
                
                line_xs = list(range(x_bound))
                line_points = []
                for x in line_xs:
                    y = m0_perp*x + c0_perp
                    if (y < 0) or (y > y_bound):
                        continue
                    line_points.append((x,y))

                pt1 = cast_tuple_to_int(line_points[0])
                pt2 = cast_tuple_to_int(line_points[-1])
                return (pt1, pt2, m0_perp, c0_perp)

            bs1 = compute_perp_bisector_in_frame(calibration_points[0], 
                                                 calibration_points[1])
            bs2 = compute_perp_bisector_in_frame(calibration_points[2], 
                                                 calibration_points[3])

            cv2.line(check_calibration_frame, bs1[0], bs1[1], (0, 0, 255), 2)                    
            cv2.line(check_calibration_frame, bs2[0], bs2[1], (0, 0, 255), 2)

            # Compute point of intersection of bisector lines, should be the arena
            # centre.
            x = (bs2[3] - bs1[3]) / (bs1[2] - bs2[2])
            y = bs1[2] * x + bs1[3]

            euc_dist = lambda p, q: np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)

            radii = [euc_dist((x,y), cp) for cp in calibration_points]

            idx = 1
            for r in radii:
                print("Radius from estimated centre to Pt #{} = {}".format(idx, r))
                idx += 1

            avg_radius = np.mean(np.array(radii))
            avg_radius_metres = (avg_radius / scale) / 1000

            print("")
            str = "These radii are measured in pixels. In theory they should be" +\
                  " identical but in practice there will be some error. If the error" +\
                  " is more than around 5-10 pixels you probably want to recalibrate."
            print(textwrap.fill(str, 60))
            print("")
            print("The average radius in metres is {:.2f}m".format(avg_radius_metres))
            print("")
            
            str = "The points you provided were used to define a circle. If the " +\
                  "frame has been correctly calibrated, this circle should overlap " +\
                  "with your arena. There will be some degree of error but it is "+\
                  "down to you to decide how accurate this needs to be."

            print(textwrap.fill(str, 60))

            cv2.circle(check_calibration_frame, cast_tuple_to_int((x,y)), int(radii[0]), (0,0,255), 2)

            points_drawn = True
            
    cv2.destroyAllWindows()

                



# if __name__ == "__main__":
#     # Use calibration info to work out homography
#     # Detect checkerboard in image then report distance between corners.

#     # Selected test video
#     video = "/home/robert/postdoc/source/output.mov"
    
#     # Chessboard parameters
#     n_rows = 6
#     n_cols = 9
#     square_size = 39

#     # Define an object chessboard image and convert to OpenCV-compatible type.
#     object_chessboard, chessboard_size =\
#           define_object_chessboard(n_rows, n_cols, square_size)
#     object_chessboard = object_chessboard.astype(np.uint8) * 255

#     refresh_cache=False
#     if refresh_cache:
#         cache_calibration_video_frames(video, 
#                                        object_chessboard, 
#                                        chessboard_size, 
#                                        N=30)
    
#     calibration = generate_calibration_from_cache(object_chessboard,
#                                                   chessboard_size,
#                                                   square_size)
    

#     # sample_image = cv2.imread('calibration_image_cache/intrinsic/000.png')
#     # imheight = sample_image.shape[0]
    
#     # border = (255 * np.ones((imheight, 100, 3))).astype(np.uint8) # generate white border
    
#     # print(sample_image.shape)

#     # dst = cv2.undistort(sample_image, 
#     #                     calibration['mtx'],
#     #                     calibration['dist'],
#     #                     None, 
#     #                     calibration['optmtx'])
    
#     # print(dst.shape)

#     # complete_frame = np.concatenate((sample_image, border, dst),  axis=1)

#     # print(calibration["rvecs"])

#     # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
#     # while cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE):
#     #     cv2.imshow('frame',complete_frame)
#     #     if cv2.waitKey(1) == 'q':
#     #         break
