import numpy as np
import cv2
import pyrealsense2 as rs
import json
import os
import sys
import argparse
from loguru import logger as lgr
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import configuration
from config.config import HandInEyeCalibConfig

# Dynamic import robot class based on configuration
from importlib import import_module
def get_robot_class_rw():
    """Dynamically imports and returns the robot class from config."""
    module = import_module("utils.arm_rw")
    return getattr(module, HandInEyeCalibConfig.robot_class_rw)
RobotRW = get_robot_class_rw()

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Hand-in-Eye Calibration')
    parser.add_argument('--ip', type=str, 
                        help='Robot IP address (e.g., ip=1.1.1.1)',
                        default=None)
    args, _ = parser.parse_known_args()
    
    for arg in sys.argv[1:]:
        if arg.startswith('ip='):
            args.ip = arg.split('=')[1]
    
    return args

CALIBRATION_BOARD_SIZE = HandInEyeCalibConfig.calibration_board_size
SQUARE_SIZE = HandInEyeCalibConfig.square_size
SAVE_DIR = HandInEyeCalibConfig.save_data_path / HandInEyeCalibConfig.exp_name
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def detect_checkerboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_BOARD_SIZE, None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        return True, corners_refined
    return False, None

def main():
    # Get IP from command line or config
    args = parse_args()
    robot_ip = args.ip if args.ip is not None else HandInEyeCalibConfig.robot_ip
    lgr.info(f"Using robot IP: {robot_ip}")

    # Initialize arm
    arm = RobotRW(robot_ip)
    
    # Initialize realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 
                         HandInEyeCalibConfig.camera_resolution[0], 
                         HandInEyeCalibConfig.camera_resolution[1], 
                         rs.format.bgr8, 
                         30)
    pipeline.start(config)

    # Main loop to capture data
    try:
        data_points = []
        save_idx = 0  # Counter for saved images
        
        while True:
            # Get frames
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            color_image = np.asanyarray(color_frame.get_data())
            
            # Detect checkerboard
            detected, corners = detect_checkerboard(color_image)
            
            # Display the image with detection status
            display_image = color_image.copy()
            if detected:
                cv2.putText(display_image, "Checkerboard detected - Press 's' to save", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No checkerboard detected", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("RealSense Camera - Press 's' to save, 'q' to quit", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if detected:
                    # Save image
                    img_path = os.path.join(SAVE_DIR, f"calib_{save_idx}.png")
                    cv2.imwrite(img_path, color_image)
                    
                    # Get robot arm pose
                    pose = arm.get_ee_pose()
                    position = pose[:3]  # XYZ (mm)
                    orientation = pose[3:]  # Euler angles (degrees)
                    
                    # Record data
                    data_points.append({
                        "image_path": img_path,
                        "position": position,
                        "orientation": orientation,
                        "corners": corners.tolist()
                    })
                    
                    lgr.success(f"Data point {save_idx} saved successfully!")
                    save_idx += 1
                else:
                    lgr.warning("Checkerboard not detected - data not saved")

    finally:
        # Save metadata
        if data_points:
            with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
                json.dump(data_points, f)
            lgr.info(f"Calibration data saved with {len(data_points)} points")
        else:
            lgr.warning("No calibration data was collected")
        
        # Close devices
        cv2.destroyAllWindows()
        pipeline.stop()

if __name__ == "__main__":
    main()