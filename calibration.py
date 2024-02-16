import cv2
import numpy as np

from time import time as timer

# create checkerboard pattern
def make_checkerboard(n_rows, n_columns, square_size):
    """
    Create checkerboard image with one pixel per millimetre of world space.

    :param n_rows: The number of rows in the pattern.
    :param n_columns: The number of columns in the pattern.
    :param square_size: The size of each checkerboard square in mm.
    :return: The checkerboard image and a tuple storing the checkerboard size (-1 in each dim)    
    """
    # Create binarised checkerboard
    rows_grid, columns_grid = np.meshgrid(range(n_rows), range(n_columns), indexing='ij')
    high_res_checkerboard = np.mod(rows_grid, 2) + np.mod(columns_grid, 2) == 1

    # Create block matrix at full resolution (1px/mm).
    square = np.ones((square_size,square_size))
    checkerboard = np.kron(high_res_checkerboard, square)

    # CV docs suggest this should be the number of inner corners
    # per dimension
    checkerboard_size = (n_columns-1, n_rows-1)


    return checkerboard, checkerboard_size

# calibration
def calib(dir, format_calibration, checkerboard, checkerboard_size):

    path = dir + 'calibration' + '.' + format_calibration

    def frame_capture(calib_frame, checkerboard, checkerboard_size):
        
        calib_obj_bw = 255 - np.uint8(checkerboard)
        ret_obj, corners_obj = cv2.findChessboardCorners(calib_obj_bw, checkerboard_size, None)

        calib_frame_gray = cv2.cvtColor(calib_frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        for i in range(1):
            # ret_img should be true on success
            ret_img, corners_img = cv2.findChessboardCorners(calib_frame_gray, checkerboard_size,  cv2.CALIB_CB_ADAPTIVE_THRESH)
            if ret_img == True:
                break
            else:
                calib_frame_gray = 255 - np.uint8(calib_frame_gray)


        return ret_img, corners_obj, corners_img

    def onChange(trackbarValue):
      cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
      cv2.imshow('frame',cap.read()[1])

    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.createTrackbar( 'capture', 'frame', 1, length, onChange )

    imgpoints = []
    objpoints = []
    extrinsic_calib_frame_id = 0

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            start = timer()
            cv2.putText(frame, 'Press p to freeze frame and view options, q to quit', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xFF == ord('p'):
                cv2.setTrackbarPos('capture','frame', int(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                cap.set(cv2.CAP_PROP_POS_FRAMES,cv2.getTrackbarPos('capture','frame'))
                frame = cap.read()[1]
                cv2.putText(frame, 'Press p to detect checkerboard corners', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                cv2.imshow('frame', frame)

                if cv2.waitKey(0) & 0xFF == ord('p'):
                    # frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.2)
                    ret_img, corners_obj, corners_img = frame_capture(frame, checkerboard, checkerboard_size)

                    if ret_img == True:
                        corners_obj = np.array(corners_obj)
                        corners_obj = corners_obj.reshape(corners_obj.shape[0],corners_obj.shape[2])
                        temp = np.zeros( (corners_obj.shape[0], corners_obj.shape[1]+1) )
                        temp[:,:-1] = corners_obj
                        corners_obj = temp

                        corners_img = np.array(corners_img)
                        corners_img = corners_img.reshape(corners_img.shape[0],corners_img.shape[2])

                        objpoints.append([corners_obj])
                        imgpoints.append([corners_img])

                        # Draw and display the corners
                        cv2.drawChessboardCorners(frame, checkerboard_size, corners_img, ret_img)
                        cv2.putText(frame, 'Press p to detect checkerboard corners', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                        cv2.putText(frame, 'Press p to proceed', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                        cv2.imshow('frame', frame)
                        frame_test = frame

                        if cv2.waitKey(0) & 0xFF == ord('p'):
                            cv2.putText(frame, 'Press p to proceed', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
                            cv2.putText(frame, 'Press o to set current frame as ground frame, spacebar to skip and proceed', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
                            cv2.imshow('frame', frame)
                            if cv2.waitKey(0) & 0xFF == ord('o'):
                                extrinsic_calib_frame_id = cap.get(cv2.CAP_PROP_POS_FRAMES)
                                frame_extrinsic = cap.read()[1]
                                # frame_extrinsic = cv2.detailEnhance(frame_extrinsic, sigma_s=10, sigma_r=0.2)
                            pass


            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break


            diff = timer() - start
            while  diff < 1/fps:
                diff = timer() - start

        else:
            cap.release()
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints = np.array(imgpoints, dtype=np.float32)

    h,  w = frame_test.shape[:2]

    # regular
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, cv2.cvtColor(frame_test, cv2.COLOR_BGR2GRAY).shape[::-1], None, None)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), cv2.CV_16SC2)
    mtx = newcameramtx
    # print(ret)

    frame_test_dst = cv2.remap(frame_test, mapx, mapy, interpolation=cv2.INTER_LINEAR)
    frame_size = (frame_test_dst.shape[1],frame_test_dst.shape[0])
    frame_size = np.array(frame_size)



    # extrinsic calibration
    id = np.array([extrinsic_calib_frame_id])
    frame_dst = cv2.remap(frame_extrinsic, mapx, mapy, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('frame', frame_dst)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

    calib_obj_bw = 255 - np.uint8(checkerboard)
    ret_obj, corners_obj = cv2.findChessboardCorners(calib_obj_bw, checkerboard_size, None)
    corners_obj = np.array(corners_obj)
    corners_obj = corners_obj.reshape(corners_obj.shape[0],corners_obj.shape[2])
    temp = np.zeros((corners_obj.shape[0],corners_obj.shape[1]+1))
    temp[:,:-1] = corners_obj
    corners_obj = temp
    corners_obj = np.float32(corners_obj)
    corners_obj = np.array([list(corners_obj)])

    ret_dst, corners_dst = cv2.findChessboardCorners(cv2.cvtColor(frame_dst, cv2.COLOR_BGR2GRAY), checkerboard_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
    corners_dst = np.squeeze(np.array(corners_dst))
    # corners_dst = corners_dst.reshape(corners_dst.shape[0],corners_dst.shape[2])
    corners_dst = np.float32(corners_dst)
    corners_dst = np.array([list(corners_dst)])
    # print(corners_dst.shape)

    scale = np.array([0.04/np.mean(np.linalg.norm(np.squeeze(corners_dst)[1:checkerboard_size[0]] - np.squeeze(corners_dst)[0:checkerboard_size[0]-1],axis=1))])
    # print(scale)

    # Homography
    H, status = cv2.findHomography(corners_dst, corners_obj)

    dst_bounds = np.transpose(np.array([[0,0,1],[frame_dst.shape[1],0,1],[frame_dst.shape[1],frame_dst.shape[0],1],[0,frame_dst.shape[0],1]]))
    map_dst_bounds = np.matmul(H,dst_bounds)

    dst_bounds = np.transpose(dst_bounds[0:2])
    dst_bounds = dst_bounds[:, np.newaxis, :]
    dst_bounds = np.float32(dst_bounds)

    map_dst_bounds = np.transpose(map_dst_bounds[0:2])
    map_dst_bounds = map_dst_bounds[:, np.newaxis, :]
    map_dst_bounds = np.float32(map_dst_bounds)

    M = cv2.getPerspectiveTransform(map_dst_bounds, dst_bounds)
    H = np.matmul(M,H)



    # save calib_data
    mapx.dump(dir + 'calib_data/' + 'mapx.dat')
    mapy.dump(dir + 'calib_data/' + 'mapy.dat')
    mtx.dump(dir + 'calib_data/' + 'mtx.dat')
    dist.dump(dir + 'calib_data/' + 'dist.dat')
    id.dump(dir + 'calib_data/' + 'id.dat')
    frame_size.dump(dir + 'calib_data/' + 'frame_size.dat')
    H.dump(dir + 'calib_data/' + 'H.dat')
    np.savetxt(dir + 'calib_data/' + 'scale' + '.csv', scale, delimiter=',')
