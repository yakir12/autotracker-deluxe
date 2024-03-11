import numpy as np
import cv2
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    # Use calibration info to work out homography
    # Detect checkerboard in image then report distance between corners.

    # Lifted from calibration dump
    n_rows = 6
    n_cols = 9
    square_size = 39 # Sq. size in mm

    object_chessboard, chessboard_size =\
          define_object_chessboard(n_rows, n_cols, square_size)
    
    # Turn it into a grayscale opencv image
    object_chessboard = object_chessboard.astype(np.uint8) * 255
        
    window = 'Test'
    board = 'Board'
    image = cv2.imread('test_image.png')

    image = undistort_image(image)

    


    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(board, cv2.WINDOW_NORMAL)

    success, corners = cv2.findChessboardCorners(image, chessboard_size)
    cv2.drawChessboardCorners(image,
                              patternSize=chessboard_size,
                              corners=corners, 
                              patternWasFound=success)
    
    object_chessboard = cv2.cvtColor(object_chessboard, cv2.COLOR_GRAY2BGR)
    success, corners = cv2.findChessboardCorners(object_chessboard, 
                                                 chessboard_size)
    cv2.drawChessboardCorners(object_chessboard,
                              patternSize=chessboard_size,
                              corners=corners, 
                              patternWasFound=success)

    cv2.imshow(window, image)
    cv2.imshow(board, object_chessboard)
    
    while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE):
        if cv2.waitKey(1) == ord('q'):
            break




