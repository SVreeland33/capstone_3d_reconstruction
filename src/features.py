import cv2
import numpy as np
from pathlib import Path
from typing import NamedTuple, Sequence

# the displayed figures get scaled down to this width so they fit on a screen
FIGURE_WIDTH = 800

class ImageFeatures(NamedTuple):
    path: Path              # which image these features came from
    keypoints: Sequence     # SIFT keypoints, position + scale + orientation
    descriptors: np.ndarray # (N, 128) SIFT descriptors, one row per keypoint

def detect_features(pathname: Path):
    ''' SIFT keypoints and descriptors for every image in pathname '''
    sift = cv2.SIFT_create()

    features = []
    for path in sorted(pathname.rglob("*")):
        # read image in gray scale
        gray = cv2.imread(str(path), 0)

        # skip anything that isn't a readable image
        if gray is None:
            continue

        keypoints, descriptors = sift.detectAndCompute(gray, None)
        features.append(ImageFeatures(path, keypoints, descriptors))

    if not features:
        raise ValueError(f'No images found in {pathname}')

    return features

def show_features(features: ImageFeatures):
    ''' checkpoint, draw the keypoints on the image - press any key for the next one '''
    img = cv2.imread(str(features.path))

    # full phone images are bigger than any screen, so scale down before drawing
    # otherwise the circles shrink with the image
    scale = FIGURE_WIDTH / img.shape[1]
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    for kp in features.keypoints:
        x, y = int(kp.pt[0] * scale), int(kp.pt[1] * scale)
        cv2.circle(img, (x, y), 2, (0, 220, 0), 1, cv2.LINE_AA)

    cv2.imshow(f'SIFT {features.path.name} - press any key for the next image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
