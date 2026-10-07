import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple, Sequence

class CalibrationResult(NamedTuple):
    views: int          # how many images actually contributed corners
    rms: float          # RMS reprojection error in pixels, lower is better
    mtx: np.ndarray     # camera matrix, the intrinsics
    dist: np.ndarray    # distortion coefficients
    rvecs: Sequence     # per-view rotation vectors
    tvecs: Sequence     # per-view translation vectors

def calibrate_camera(pathname: Path, pattern_size=(9, 6)):
    # Termination criteria for ending our algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points for reading
    cols, rows = pattern_size
    objp = np.zeros((cols*rows, 3), np.float32)
    objp[:,:2] = np.mgrid[0:cols,0:rows].T.reshape(-1,2)

    obj_points = [] # 3D points
    img_points = [] # 2D points

    # read all images from pathname
    image_size = None
    for path in sorted(pathname.rglob("*")):
        img = cv2.imread(str(path))

        # skip anything that isn't a readable image
        if img is None:
            continue

        # Convert image color to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # calibrateCamera wants (width, height), shape gives (height, width)
        image_size = gray_img.shape[::-1]

        # Get corners of chessboard
        ret, corners = cv2.findChessboardCorners(gray_img, pattern_size, None)

        # Check if corners were found and add object and image points
        if ret == True:
            obj_points.append(objp)

            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners2)

    # nothing detected means nothing to calibrate from
    if image_size is None or not img_points:
        raise ValueError(f'No chessboard corners found in {pathname}')

    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

    return CalibrationResult(len(img_points), rms, mtx, dist, rvecs, tvecs)
