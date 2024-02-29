import cv2
import numpy as np

from random import randint

import pandas as pd
import os

def autotracker(dir, 
                track_filename, 
                label, 
                format_track, 
                desired_tracker,
                working_csv='raw_tracks.csv'):
    """
    :param dir: Full path to data directory (e.g. data/<uname>/<session>/)
    :param track_filename: Calibrated video file for tracking.
    :param label: Session ID
    :param format_track: Calibrated video file format
    :param desired_tracker: Tracker parameter passed to OpenCV
    :param working_csv: The filename of the csv file in which to store track
                        data.
    """
    input_dir = dir + track_filename + "." + format_track
    working_csv = dir + working_csv

    cap = cv2.VideoCapture(input_dir)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    window_name = 'autotrack'
    trackbar_name = 'capture'    

    # Trackbar callback
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


    # Extract greyscale background frame
    bg_frame = compute_background_frame(cap, 
                                        method='first_N_median',
                                        N=10)
    bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

    #
    # State variables
    #
    tracking = False
    bbox = None
    centroid = np.nan
    centroid_track = []
        
    while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE):
        success, clean_frame = cap.read()
        tracking_status_str = "(TRACKING) " if tracking else ""

        if success:
            # Frame and trackbar update
            if (not (bbox==None)) and (not (centroid == np.nan)):
                # If centroid defined then we are tracking.
                frame_centroid = (int(bbox[0] + centroid[0]),
                                  int(bbox[1] + centroid[1]))

                # Draw tracked point on clean frame. Clean frame used 
                # specifically so that pause behaviour is intuitive (track 
                # point still shown when paused).
                cv2.circle(clean_frame,
                           frame_centroid,
                           radius=5,
                           color=(0, 0, 255),
                           thickness=cv2.FILLED)
                

            frame = clean_frame.copy()
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
                    pause_frame = clean_frame.copy()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 
                            cv2.getTrackbarPos(trackbar_name, window_name))
                    
                    tracking_context_str = 'begin'
                    tracking_status_str = ''
                    if tracking:
                        tracking_status_str = '(TRACKING)'
                        tracking_context_str = 'end'

                    write_on_frame(pause_frame, 
                                  '{} Press p to resume, t to {} track'
                                  .format(tracking_status_str,
                                          tracking_context_str)
                                   )                
                    cv2.imshow(window_name, pause_frame)
                    kp = cv2.waitKey(1)

                    if kp == ord('p'):
                        break # Break to main play loop
                    elif kp == ord('t'):
                        # Tracking init
                        if not tracking:
                            bbox = cv2.selectROI('Select ROI',
                                                clean_frame, 
                                                fromCenter=False, 
                                                showCrosshair=True)
                        
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
                                bbox = None
                        else:
                            # Save tracks to sane format here!
                            tracking = False
                            bbox = None
                            centroid = np.nan 

                            write_track_to_file(centroid_track, working_csv)

                            
            elif kp == ord('q'):
                break

            #
            # Perform tracking
            #
            if tracking and (not (bbox == None)):
                # For debugging
                roi_window = 'ROI'
                cv2.namedWindow(roi_window,cv2.WINDOW_NORMAL)

                track_success, bbox = tracker.update(clean_frame)

                if not track_success:
                    print("FAILED")

                frame_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(frame_gray, bg_gray)
                _, mask_frame = cv2.threshold(diff, 
                                              25, 
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
                else:
                    centroid = np.nan
                
                cv2.imshow(roi_window,bin_roi)

                frame_centroid = (int(bbox[0] + centroid[0]),
                                  int(bbox[1] + centroid[1]))
                centroid_track.append(frame_centroid)

            elif tracking and bbox == None:
                print("Bounding box is undefined, somehow tracking has"+
                      " been started without a defined bbox.")
                print("How did you manage this?")
                tracking = False # Quietly disable tracking


    if tracking:
        write_track_to_file(centroid_track, working_csv)
        print("WARNING: You quit while tracking. The last track has been" + 
              " saved up to the point where you closed the window.")

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
    else:
        print("Background construction method ({}) not recognised."
              .format(method))
        print("Defaulting to 'first_N_median'")
        return compute_background_frame(capture, method='first_N_median', N=10)
    
    # Reset capture to beginning
    capture.set(cv2.CAP_PROP_POS_FRAMES, current_capture_position)
    return background_frame


def write_track_to_file(points, filepath):
    """
    Write a track out to a csv file.

    :param points: The 2D points which make up the track.
    :param filepath: The filepath for storage.
    """
    # Generate track file if it doesn't exist
    if not os.path.isfile(filepath):
        pd.DataFrame().to_csv(filepath)

    df = pd.read_csv(filepath)
    columns = df.columns.to_list()

    if len(columns) == 1:
        # A new file will have one column called 'Unnamed'                
        track_idx = 0
    else:
        # Column names are 'track_i_d' where i is the 
        # index of the track and d is the dimension
        # (x or y)
        track_idx = int(columns[-1].split("_")[1]) + 1
    
    x_col = 'track_{}_x'.format(track_idx)
    y_col = 'track_{}_y'.format(track_idx)
    xs = [x for (x,_) in points]
    ys = [y for (_,y) in points]

    new_cols = pd.DataFrame(columns=[x_col, y_col])
    new_cols.loc[:, x_col] = xs
    new_cols.loc[:, y_col] = ys
    print(new_cols)

    # NOT YET WORKING, need to implement ragged join
    # df = df.join(new_cols)
    # print(df)
    # df.to_csv(filepath)

    # print("Track {} added to {}.".format(track_idx, filepath))
