from pathlib import Path

from calibration import calibrate_camera
from features import detect_features, show_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    calibration_path = PROJECT_ROOT / "images" / "calibration_images"
    object_path = PROJECT_ROOT / "images" / "controller"

    calib = calibrate_camera(calibration_path)

    # feature detection
    features = detect_features(object_path)

    # checkpoint, look at what SIFT found in each image
    for image_features in features:
        print(f'{image_features.path.name}: {len(image_features.keypoints)} keypoints')
        show_features(image_features)

if __name__ == "__main__":
    main()
