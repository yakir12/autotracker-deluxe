import cv2
import numpy as np

from random import randint

# autotracker
def autotracker(dir, track_filename, label, format_track, desired_tracker):
    input_dir = dir + track_filename + '.' + format_track
    cap = cv2.VideoCapture(input_dir)
    fps = cap.get(cv2.CAP_PROP_FPS)
    split_time = np.array([])   # split trajectory at these times - separate rolls

    def nothing(x):
      pass

    # capture start and end frames
    def onChange_start(trackbarValue):
      cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
      start_img = cap.read()[1]
      cv2.putText(start_img, 'Slide trackbar to select start frame for tracking', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
      cv2.imshow('Start', start_img)


    def onChange_end(trackbarValue):
      cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
      end_img = cap.read()[1]
      cv2.putText(end_img, 'Slide trackbar to select end frame for tracking', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
      cv2.imshow('End', end_img)


    while True:
        cv2.namedWindow('Start', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Start', 700, 500)
        cv2.moveWindow('Start', 10, 100)
        cv2.namedWindow('End', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('End', 700, 500)
        cv2.moveWindow('End', 750, 100)
        cv2.createTrackbar( 'start', 'Start', 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), onChange_start )
        cv2.createTrackbar( 'end'  , 'End', int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), onChange_end )
        cv2.waitKey()

        start = cv2.getTrackbarPos('start','Start')
        end   = cv2.getTrackbarPos('end','End')

        if start >= end:
            print("start must be less than end")
            cv2.destroyAllWindows()
        else:
            break

    cv2.destroyAllWindows()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    success, frame = cap.read()


    if success:

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
          tracker = cv2.legacy.TrackerGOTURN_createlength()
      elif desired_tracker == 'MOSSE':
          tracker = cv2.legacy.TrackerMOSSE_create()
      elif desired_tracker == 'CSRT':
          tracker = cv2.legacy.TrackerCSRT_create()
      else:
          tracker = None
          print('The name of the tracker is incorrect')

      # Set bounding box drawing parameters
      from_center = False
      show_cross_hair = False
      cv2.namedWindow('Object Tracker', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Object Tracker', 700, 500)
      cv2.putText(frame, 'Draw a box around beetle+ball', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
      bounding_box = cv2.selectROI('Object Tracker', frame, from_center, show_cross_hair)
      color_list = (randint(127, 250), randint(127, 250), 0)
      cv2.destroyAllWindows()

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    success, frame = cap.read()

    # capture start and end frames for cut
    def onChange_startcut(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        startcut_img = cap.read()[1]
        cv2.putText(startcut_img, 'Slide trackbar to select frame to begin cut', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('Start cut', startcut_img)

    def onChange_endcut(trackbarValue):
        cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
        endcut_img = cap.read()[1]
        cv2.putText(endcut_img, 'Slide trackbar to select frame to end cut', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow('End cut', endcut_img)

    if success:

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
            print('The name of the tracker is incorrect')

        # Set bounding box drawing parameters
        from_center = False
        show_cross_hair = False

        # bounding_box = cv2.selectROI('Object Tracker', frame, from_center, show_cross_hair)
        color_list = (randint(127, 250), randint(127, 250), 0)
        # cv2.destroyAllWindows()

        cap_temp = cap
        frames_temp = []

        for i in range(10):
            cap_temp.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_POS_FRAMES)-i, 0))
            ret, frame_temp = cap_temp.read()
            frames_temp.append(frame_temp)

        medianFrame = np.median(frames_temp, axis=0).astype(dtype=np.uint8)

        cap = cv2.VideoCapture(input_dir)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        medianFrame_gray = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        diff_frame = cv2.absdiff(frame_gray, medianFrame_gray)
        th, diff_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)
        frame_mask = cv2.bitwise_and(frame, frame, mask=diff_frame)

        ROI_init = frame[int(bounding_box[1]):int(bounding_box[1])+int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[0])+int(bounding_box[2])]
        tracker.init(frame, bounding_box)

        t_stamp = []
        traj = []
        phi = []

        # create VideoWriter object to save video output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        result = cv2.VideoWriter(dir + 'processed' + '_autotrack' + '.' + 'mp4', fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Tracking', 700, 500)
        cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
        # cv2.resizeWindow('ROI', 30, 30)

        # process video
        while cap.isOpened():

          # cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)
          # curr_pos +=1

          if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
              break

          success, frame = cap.read()
          flag = 0

          if success:
            track_success, bbox = tracker.update(frame)
            point_1 = (int(bbox[0]), int(bbox[1]))
            point_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            c = (np.array(point_1) + np.array(point_2))/2

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_frame = cv2.absdiff(frame_gray, medianFrame_gray)
            th, diff_frame = cv2.threshold(diff_frame, 25, 255, cv2.THRESH_BINARY)
            frame_mask = cv2.bitwise_and(frame, frame, mask=diff_frame)
            ROI = frame_mask[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])]

            ROI_gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
            contours,hierarchy = cv2.findContours(ROI_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            if len(contours) != 0:
                for i in range(len(contours)):
                    if cv2.contourArea(contours[i]) >= 30 and cv2.contourArea(contours[i]) <= 600:
                        ell = cv2.fitEllipseDirect(contours[i])
                        cv2.ellipse(ROI,ell,color_list,1)
                        phi.append((cap.get(cv2.CAP_PROP_POS_FRAMES)/fps,ell[2]))
                        c = np.array(point_1) + np.array(ell[1])
                        flag = 1

            # pause to reinitialize if tracker fails
            if cv2.waitKey(1) & 0xFF == ord('p'):
                curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cv2.putText(frame, 'Press p to pause', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (1,1,1), 2, cv2.LINE_AA)
                cv2.putText(frame, 'Press o if cue switch, spacebar to trim if tracker has failed mid-roll, p to skip and continue', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow("Tracking", frame)
                key_press = cv2.waitKey(0) & 0xFF

                if key_press == ord('p'):
                    curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    cap_temp = cap
                    frames_temp = []

                    for i in range(10):
                        cap_temp.set(cv2.CAP_PROP_POS_FRAMES, max(cap.get(cv2.CAP_PROP_POS_FRAMES)-fps-i, 0))
                        ret, frame_temp = cap_temp.read()
                        frames_temp.append(frame_temp)

                    medianFrame = np.median(frames_temp, axis=0).astype(dtype=np.uint8)
                    medianFrame_gray = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos)

                elif key_press == ord('o'):
                    # tracker.clear()
                    cv2.destroyAllWindows()

                    cv2.namedWindow('Start cut', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Start cut', 700, 500)
                    cv2.moveWindow('Start cut', 10, 100)
                    cv2.namedWindow('End cut', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('End cut', 700, 500)
                    cv2.moveWindow('End cut', 750, 100)
                    cv2.createTrackbar( 'start', 'Start cut', 1, int(curr_pos), onChange_startcut )
                    cv2.createTrackbar( 'end'  , 'End cut', int(curr_pos), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), onChange_endcut )
                    cv2.waitKey()

                    start_cut = cv2.getTrackbarPos('start','Start cut')
                    #end_cut   = cv2.getTrackbarPos('end','End cut')
                    end_cut = start_cut

                    n_delete = int(curr_pos) - start_cut
                    t_stamp = t_stamp[: len(t_stamp)-n_delete]
                    traj = traj[: len(traj)-n_delete]

                    split_time = np.append(split_time, t_stamp[-1])

                    cv2.destroyAllWindows()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, end_cut)
                    frame = cap.read()[1]
                    cv2.namedWindow('Object Tracker', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Object Tracker', 700, 500)
                    cv2.putText(frame, 'Draw a box around beetle+ball', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    bbox = cv2.selectROI('Object Tracker', frame, from_center, show_cross_hair)
                    cv2.destroyWindow('Object Tracker')
                    newtracker = cv2.legacy.TrackerBoosting_create()
                    newtracker.init(frame, bbox)
                    tracker = newtracker


                else:
                    # tracker.clear()
                    cv2.destroyAllWindows()

                    cv2.namedWindow('Start cut', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Start cut', 700, 500)
                    cv2.moveWindow('Start cut', 10, 100)
                    cv2.namedWindow('End cut', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('End cut', 700, 500)
                    cv2.moveWindow('End cut', 750, 100)
                    cv2.createTrackbar( 'start', 'Start cut', 1, int(curr_pos), onChange_startcut )
                    cv2.createTrackbar( 'end'  , 'End cut', int(curr_pos), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), onChange_endcut )
                    cv2.waitKey()

                    start_cut = cv2.getTrackbarPos('start','Start cut')
                    end_cut   = cv2.getTrackbarPos('end','End cut')

                    n_delete = int(curr_pos) - start_cut
                    t_stamp = t_stamp[: len(t_stamp)-n_delete]
                    traj = traj[: len(traj)-n_delete]

                    cv2.destroyAllWindows()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, end_cut)
                    frame = cap.read()[1]
                    cv2.namedWindow('Object Tracker', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('Object Tracker', 700, 500)
                    cv2.putText(frame, 'Draw a box around beetle+ball', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                    bbox = cv2.selectROI('Object Tracker', frame, from_center, show_cross_hair)
                    cv2.destroyWindow('Object Tracker')
                    newtracker = cv2.legacy.TrackerBoosting_create()
                    newtracker.init(frame, bbox)
                    tracker = newtracker

            cv2.rectangle(frame,point_1,point_2,(0,0,255),1)
            result.write(frame)

            # if flag == 0:
                # c = np.array(traj)[-1]

            t_stamp.append(cap.get(cv2.CAP_PROP_POS_FRAMES)/fps)
            traj.append(c)

            cv2.namedWindow('Tracking', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Tracking', 700, 500)
            cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
            # cv2.resizeWindow('ROI', 50, 30)
            cv2.putText(frame, 'Press p to pause', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Tracking", frame)
            cv2.imshow("ROI", ROI)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # no more video frames left
          else:
            break

        cv2.destroyAllWindows()

        # post-processing
        traj = np.array(traj, dtype="float")
        traj = np.transpose(traj)

        t_stamp = np.array(t_stamp, dtype="float")

        phi = np.array(phi, dtype='float')

        fps_save = []
        fps_save.append(fps)
        fps = np.array(fps_save, dtype='float')

        s1 = np.array(1920)
        s2 = np.array(1080)
        traj_0 = traj[0]/s1
        traj_1 = 1 - traj[1]/s2

        np.savetxt(dir + 'tracking_data/' + 'coordinates' + '_' + label + '.csv', (traj_0, traj_1), delimiter=',')
        np.savetxt(dir + 'tracking_data/' + 'heading' + '_' + label + '.csv', phi, delimiter=',')
        np.savetxt(dir + 'tracking_data/' + 'fps' + '_' + label + '.csv', fps, delimiter=',')
        np.savetxt(dir + 'tracking_data/' + 'time' + '_' + label + '.csv', t_stamp, delimiter=',')
        np.savetxt(dir + 'tracking_data/' + 'track_split' + '_' + label + '.csv', split_time, delimiter=',')

        cap.release()
        result.release()
