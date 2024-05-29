import cv2
import numpy as np
import copy

from random import randint

import pandas as pd
import os

from dtrack_params import dtrack_params
from project import project_file

def get_video_file_extension(directory, filename):
    """
    Assuming there is a videofile in the working directory with name of the form
    <filename>.<format>, this function will return <format> given <filename>.

    :param directory: The directory in which to look
    :param filename: <filename> as above.
    :return: The file format as a string without the '.'
    """
    entries = os.listdir(directory)

    # Look for all files where 'filename' forms the complete segment before the
    # period.
    matching = [e for e in entries if e.split('.')[0] == filename]

    # All filenames are handled internally so if this assertion fails it 
    # implies a file has been created which breaks the unique filename 
    # assumption of this function. (i.e. this is a bug, not user error.)
    assert(len(matching) == 1)

    # Decompose the matching entry and check it has a second half. The 'split'
    # filtering used above will succeed even if the string does not contain a
    # '.' delimeter. Again, failure here indicates a bug.
    components = matching[0].split('.')
    assert(len(components) == 2)
    
    # Return the file extension
    return components[1]
    

def autotracker():
    """
    Spawns an OpenCV window which plays the video for tracking. The user can 
    pause and begin tracking. Tracking is handled by an OpenCV tracker (currently
    BOOSTING). The actual point which is tracked is the centre of the bounding box
    (so we rely on the tracker being accurate).
    """

    input_dir = project_file["tracking_video"]
    project_directory = dtrack_params["project_directory"]
    
    desired_tracker = dtrack_params["options.autotracker.cv_backend"]
    track_point = dtrack_params["options.autotracker.track_point"]
    bg_computation_method = dtrack_params["options.autotracker.bg_computation_method"]
    bg_sample_size = dtrack_params["options.autotracker.bg_sample_size"]
    track_interval = dtrack_params["options.autotracker.track_interval"]
    show_roi = dtrack_params["options.autotracker.show_roi"]
    
    cap = cv2.VideoCapture(input_dir)

    # Store FPS for analysis stage
    project_file["track_fps"] = cap.get(cv2.CAP_PROP_FPS)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_name = 'autotrack'
    trackbar_name = 'capture'    

    frame_idx = 0

    # Trackbar callback, if chosen frame is not a multiple of the
    # tracking interval, then jump to next frame which is.
    trackbar_external_callback = False
    def tb_callback(trackbar_value):
        if trackbar_external_callback:
            cap.set(cv2.CAP_PROP_POS_FRAMES, trackbar_value)
            cv2.imshow(window_name, cap.read()[1])


    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(trackbar_name, window_name, 1, length, tb_callback)
    print("Input file: {}".format(input_dir))
    
    tracker = get_tracker_from_string(desired_tracker)
    if tracker == None:
        print("Chosen tracker ({}) not recognised, exiting autotracker."              
              .format(tracker) + " Check main.py configuration.")
        return

    print("Chosen tracker: {}".format(desired_tracker))

    # Extract background frame. 
    # A background frame is always computed but if tracking is using the centre
    # of the bbox, then the background subtration isn't used.
    bg_hsv = compute_background_frame(cap,
                                      method=bg_computation_method,
                                      N=bg_sample_size)
    
    # Background subtraction is performed in HSV colour space. Not sure if
    # there's a difference in computing the bcackground frame in HSV as opposed
    # to computing it in BGR then converting to HSV.
    bg_v = cv2.cvtColor(bg_hsv, cv2.COLOR_BGR2HSV)[:,:,2]
    
    #
    # State variables
    #
    tracking = False
    segmentation_success = False
    bbox = None
    
    centroid = np.nan
    centroid_track = []

    timestamps = []

    # Carry first defined bbox forward onto multiple tracks.
    assume_bbox = dtrack_params["options.autotracker.remember_roi"]
    first_bbox = None

    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        success, clean_frame = cap.read()
        frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)

        if (frame_idx % track_interval) != 0:
            continue
        
        
        display_frame = clean_frame.copy()
        tracking_status_str = "(TRACKING) " if tracking else ""

        if success:
            # Frame and trackbar update
            if (not (bbox==None)) and (not np.isnan(centroid).any()):
                # If centroid defined then we are tracking.
                frame_centroid = (int(bbox[0] + centroid[0]),
                                  int(bbox[1] + centroid[1]))


                colour = (0,255,0)
                if not segmentation_success:
                    colour = (0,0,255)

                # Draw tracked point on clean frame. Clean frame used 
                # specifically so that pause behaviour is intuitive (track 
                # point still shown when paused).
                cv2.circle(display_frame,
                           frame_centroid,
                           radius=5,
                           color=colour,
                           thickness=cv2.FILLED)
                
                cv2.rectangle(display_frame,
                              pt1=(int(bbox[0]), int(bbox[1])),
                              pt2=(int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              color=colour,
                              thickness=3
                             )
            
            frame = display_frame.copy()
            
            write_on_frame(frame, 
                           '{} Press p to pause and view options, q to quit.'
                           .format(tracking_status_str))

            
            cv2.imshow(window_name, frame)
            trackbar_external_callback = False
            cv2.setTrackbarPos(trackbar_name, 
                               window_name, 
                               int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
            trackbar_external_callback = True


            #
            # Loop control logic, pause, start tracking, select ROI, 
            # end tracking
            # 
            kp = cv2.waitKey(1)
            if kp == ord('p'):
                # Pause
                while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 
                            cv2.getTrackbarPos(trackbar_name, window_name))
                    success, pause_frame = cap.read()
                    
                    tracking_context_str = 'begin'
                    tracking_status_str = ''
                    if tracking:
                        tracking_status_str = '(TRACKING)'
                        tracking_context_str = 'end'

                    assume_bbox_string = ""
                    if assume_bbox:
                        assume_bbox_string = ", or r to (re)define ROI."

                    write_on_frame(pause_frame, 
                                  '{} Press p to resume, t to {} track{}'
                                  .format(tracking_status_str,
                                          tracking_context_str,
                                          assume_bbox_string)
                                   )                
                    cv2.imshow(window_name, pause_frame)
                    kp = cv2.waitKey(1)

                    if kp == ord('p'):
                        break # Break to main play loop
                    elif kp == ord('r'):
                        old_bbox = first_bbox
                        first_bbox = cv2.selectROI('Select ROI',
                                                clean_frame, 
                                                fromCenter=True, 
                                                showCrosshair=True)
                        if first_bbox != (0,0,0,0):
                            bbox = first_bbox
                        else:
                            # If user cancels, restore old bbox.
                            first_bbox = old_bbox
                            bbox = first_bbox
                    elif kp == ord('t'):
                        # Tracking init
                        if not tracking:
                            if assume_bbox and (not first_bbox == None):
                                bbox = first_bbox
                            else:
                                first_bbox = cv2.selectROI('Select ROI',
                                                    clean_frame, 
                                                    fromCenter=True, 
                                                    showCrosshair=True)
                                bbox = first_bbox                            
                               
                            
                            # Bbox is cv Rect, tuple (x, y, width, height) where
                            # x/y measured from top left of frame.
                            if not (bbox == (0,0,0,0)):
                                tracking = True
                                tracker = get_tracker_from_string(desired_tracker)
                                tracker.init(clean_frame, bbox)
                            else:
                                # Unset any tracking variables. Overwrite kp
                                # to stop ROI selection from popping up 
                                # constantly.
                                kp = ord('z') 
                                centroid = np.nan
                                first_bbox = None
                                bbox = None
                        else:
                            # Write last track to file
                            write_track_and_time_to_file(centroid_track, 
                                                         timestamps, 
                                                         project_directory)
                            
                            # Reset online tracking info.
                            tracking = False
                            bbox = None
                            centroid = np.nan 
                            centroid_track = []
                            timestamps = []

                            
            elif kp == ord('q'):
                break

            #
            # Perform tracking
            #
            if tracking and (not (bbox == None)):


                track_success, bbox = tracker.update(clean_frame)

                if not track_success:
                    print("FAILED")

                frame_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                frame_hsv = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2HSV)
                frame_v = frame_hsv[:,:,2]

                diff = cv2.absdiff(frame_v, bg_v)
                _, mask_frame = cv2.threshold(diff, 
                                              15, 
                                              255,
                                              cv2.THRESH_BINARY)


                #
                # Old verison, pulled out a masked grayscale ROI and then
                # found contours in this frame. This is equivalent to finding
                # contours in a binary image. Using binary image may be
                # advantageous as we (1) don't need to create extra frames
                # and (2) can apply erosion and dilation to denoise the ROI.
                # 
                masked_frame = cv2.bitwise_and(clean_frame, 
                                               clean_frame, 
                                               mask=diff)

                # Slice region of interest out of masked frame.
                # roi = masked_frame[int(bbox[1]):int(bbox[1] + bbox[3]),
                #                    int(bbox[0]):int(bbox[0] + bbox[2])]
                # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)                
                
                # Slice region of interst out of binary frame
                bin_roi = mask_frame[int(bbox[1]):int(bbox[1] + bbox[3]),
                                     int(bbox[0]):int(bbox[0] + bbox[2])]

                # May want to erode/dilate here.  
                morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                                         (5,5))
                bin_roi = cv2.erode(bin_roi, kernel=morph_kernel)
                bin_roi = cv2.dilate(bin_roi, kernel=morph_kernel, iterations=2)

                roi_moments = cv2.moments(bin_roi, binaryImage=True)
                m00 = roi_moments["m00"]
                m10 = roi_moments["m10"]
                m01 = roi_moments["m01"]

                if not (m00 == 0):
                    centroid = (int(m10/m00), int(m01/m00))
                    cv2.circle(bin_roi, 
                            centroid, 
                            radius=3, 
                            color=(0,0,255),
                            thickness=cv2.FILLED)
                    segmentation_success = True
                else:
                    # If no area, set tracking to centre of bbox
                    segmentation_success = False
                    centroid = (bbox[2]/2, bbox[3]/2)

                # Use bbox centre as track point instead of trying to segment
                # the beetle out of the region of interest.
                if track_point == "centre-of-bounding-box":
                    segmentation_success = False
                    centroid = (bbox[2]/2, bbox[3]/2)                    
                
                if show_roi:
                    roi_window = 'ROI'
                    cv2.namedWindow(roi_window,cv2.WINDOW_NORMAL)
                    cv2.imshow(roi_window, bin_roi)

                frame_centroid = (int(bbox[0] + centroid[0]),
                                  int(bbox[1] + centroid[1]))
                centroid_track.append(frame_centroid)
                ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                
                timestamps.append(ts)

            elif tracking and bbox == None:
                print("Bounding box is undefined, somehow tracking has"+
                      " been started without a defined bbox.")
                print("How did you manage this?")
                tracking = False # Quietly disable tracking


    if tracking:
        print("WARNING: You quit while tracking. Attempting to save final track...")
        write_track_and_time_to_file(centroid_track, timestamps, project_directory)
        

    cv2.destroyAllWindows()
    cap.release()
    



def write_on_frame(frame, text):
    """
    Helper to write text onto an OpenCV frame. If the frame already has text
    on it, new text will be overlayed, so keep a clean copy of the frame.

    Note, this method modifies the frame passed in-place. No copies are made.

    :param frame: The clean OpenCV frame.
    :param text: The text to add.
    """
    cv2.putText(frame, 
                text,
                (50,50), # Origin
                cv2.FONT_HERSHEY_SIMPLEX, # Font 
                1, # Font scale
                (255,255,255), # colour
                2, # Line thickness
                cv2.LINE_AA # Line type
                )                
    

def get_tracker_from_string(desired_tracker):
    """
    Containment method to sequester 'desired tracker' selection.

    :param desired_tracker: The string tracker identifier passed to autotracker.
    :return: cv2 tracker object or None if string did not identify a tracker.
    """
    tracker = None
    if desired_tracker == 'BOOSTING':
        tracker = cv2.legacy.TrackerBoosting_create()
    elif desired_tracker == 'MIL':
        tracker = cv2.legacy.TrackerMIL_create()
    elif desired_tracker == 'KCF':
        tracker = cv2.legacy.TrackerKCF_create()
    elif desired_tracker == 'TLD':
        tracker = cv2.legacy.TrackerTLD_create()
    elif desired_tracker == 'MEDIANFLOW':
        tracker = cv2.legacy.TrackerMedianFlow_create()
    elif desired_tracker == 'GOTURN':
        tracker = cv2.legacy.TrackerGOTURN_create()
    elif desired_tracker == 'MOSSE':
        tracker = cv2.legacy.TrackerMOSSE_create()
    elif desired_tracker == 'CSRT':
        tracker = cv2.legacy.TrackerCSRT_create()
    else:
        tracker = None
    return tracker


def compute_background_frame(capture, method='first_N_median', N=10):
    """
    Compute a background frame for a given capture using the specified method.

    For the method option:
    'first_N_median' takes the median of the first N frames 
    'first_N_mean' takes the mean of the first N frames
    'first_N_median_HSV' As first_N_median but in HSV colourspace
    
    'random_N_mean' Takes the mean of a random sample of N frames from the video.
    This meathod is slow and should only be used to pre-compute a background.

    N = 10 by default
    
    :param capture: The OpenCV VideoCapture object.
    :param method: The method used to construct a background frame. Supported
                   are 'first_N_median' and 'first_N_mean'
    :param N: The number of frames to use for a given method.                   
    """

    # Store capture position before modification
    current_capture_position = capture.get(cv2.CAP_PROP_POS_FRAMES)

    if method == 'first_N_median':
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)          
        background_sample = []
        for i in range(N):
            success, frame = capture.read()
            background_sample.append(frame)
        
        background_frame =\
              np.median(background_sample, axis=0).astype(dtype=np.uint8)
    elif method == 'first_N_mean':
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)          
        background_sample = []
        for i in range(N):
            success, frame = capture.read()
            background_sample.append(frame)
        
        background_frame =\
              np.mean(background_sample, axis=0).astype(dtype=np.uint8)
        
    elif method == 'first_N_median_HSV':
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)          
        background_sample = []
        for i in range(N):
            success, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            background_sample.append(frame)
        
        background_frame =\
              np.median(background_sample, axis=0).astype(dtype=np.uint8)        
        
    elif method == 'random_N_mean':
        # This method is slow, do not use it in real time.

        # Create RandomState for repeatability, should be configurable/refreshable
        # in future.
        random_state = np.random.RandomState(seed=493570483)

        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        
        if N > frame_count:
            # If N is too big, set N to frame count.
            N = frame_count 

        # Generate a series of indices up to the maximum frame index and
        # choose N indices from this set without replacement (random subset
        # of frame indices).
        sample_indices = random_state.choice(range(int(frame_count)), 
                                            size=int(N), 
                                            replace=False)

        # Initialise mean to first of sample frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, sample_indices[0])
        success, frame = capture.read()
        mean_frame = frame

        print('Computing mean background frame...') 
        counter = 1
        for idx in range(1, len(sample_indices)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            success, frame = capture.read()

            # Compute mean frame as rolling average
            # mean = (mean*(n-1) + x) / n
            mean_frame = (mean_frame*counter + frame) / (counter + 1)

            counter += 1 # Increment counter

        background_frame = mean_frame.astype(np.uint8)

    else:
        print("Background construction method ({}) not recognised."
              .format(method))
        print("Defaulting to 'first_N_median'")
        return compute_background_frame(capture, method='first_N_median', N=10)
    
    # Reset capture to beginning
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_capture_position)
    return background_frame


def write_track_and_time_to_file(points, timestamps, basepath):
    """
    Write a track out to a csv file.

    :param points: The 2D points which make up the track.
    :param timestamps: The timestamp (in milliseconds) for each track point
    :param basepath: The location to store the track files (project directory)
    """

    # Should have one timestamp for each trackpoint
    assert(len(points) == len(timestamps))

    # Zero-length tracks or single points aren't allowed.
    if len(points) < 2:
        print("Track has < 2 points and will not be stored.")
        return

    trackpath = os.path.join(basepath, 'raw_tracks.csv')
    timepath = os.path.join(basepath, 'timestamps.csv')

    # If file doesn't exist, we need to create it
    new_file = not os.path.isfile(trackpath)
    if new_file:
        # Column names are 'track_i_d' where i is the 
        # index of the track and d is the dimension
        # (x or y)
        x_label = 'track_0_x'
        y_label = 'track_0_y'
        df = pd.DataFrame(columns=[x_label, y_label])
        df.loc[:, x_label] = [x for (x,_) in points]
        df.loc[:, y_label] = [y for (_,y) in points]        
        df.to_csv(trackpath)
        print("New track file created at {}".format(trackpath))

        time_label = 'track_0'
        df = pd.DataFrame(columns=[time_label])
        df.loc[:, time_label] = timestamps
        df.to_csv(timepath)
        print("New timestamp file created at {}".format(timepath)) 
        return 

    #
    # Otherwise, open file, determine track number, and write new data
    # to dataframe.
    #
    df = pd.read_csv(trackpath, index_col=[0])
    columns = df.columns.to_list()
    track_idx = int(columns[-1].split("_")[1]) + 1
    
    x_col = 'track_{}_x'.format(track_idx)
    y_col = 'track_{}_y'.format(track_idx)
    xs = [x for (x,_) in points]
    ys = [y for (_,y) in points]

    new_cols = pd.DataFrame(columns=[x_col, y_col])
    new_cols.loc[:, x_col] = xs
    new_cols.loc[:, y_col] = ys

    # Add new columns to main dataframe.
    df = ragged_join(df,new_cols)
    df.to_csv(trackpath)

    print("Track {} added to {}.".format(track_idx, trackpath))

    timefile_exists = os.path.exists(timepath)
    if not timefile_exists:
        time_label = 'track_{}'.format(track_idx)
        df = pd.DataFrame(columns=[time_label])
        df.loc[:, time_label] = timestamps
        df.to_csv(timepath)
        print("New timestamp file created at {}, starting from Track {}".format(timepath, track_idx)) 
        return
    
    time_df = pd.read_csv(timepath, index_col=[0])
    track_col = 'track_{}'.format(track_idx)
    new_col = pd.DataFrame(columns=[track_col])
    new_col.loc[:,track_col] = timestamps
    df = ragged_join(time_df, new_col)
    df.to_csv(timepath)

    print("Timestamps stored for track {}".format(track_idx))

def ragged_join(dfA, dfB):
    """
    Implements a ragged horizontal join for dataframes.

    Where two dataframes have different lengths of index, this function will
    replace the index of the shorter dataframe with that of the longer one, then
    join the two dataframes and return the new complete dataframe.

    :param dfA: The main dataframe to which you want to add columns
    :param dfB: The additional columns (also a DataFrame)
    """
    # If A longer, reindex B and join onto A
    if len(dfA.index) > len(dfB.index):
        return dfA.join(dfB.reindex(index=dfA.index))
    
    # If B longer, reindex A and join B onto A.
    return dfA.reindex(index=dfB.index).join(dfB)

