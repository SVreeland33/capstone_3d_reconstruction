from pathlib import Path

from calibration import calibrate_camera

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    images_path = PROJECT_ROOT / "images" / "calibration_images"

    calib = calibrate_camera(images_path)

    # Next steps: feature detection, matching, pose estimation, triangulation, visualization

if __name__ == "__main__":
    main()
