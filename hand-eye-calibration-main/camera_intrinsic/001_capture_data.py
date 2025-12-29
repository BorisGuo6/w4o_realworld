import numpy as np
import cv2
import pyrealsense2 as rs
import os
import sys
import pickle
from pathlib import Path
from loguru import logger as lgr

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import configuration
from config.config import CameraIntrinsicCalibConfig

# Configuration parameters
CALIBRATION_BOARD_SIZE = CameraIntrinsicCalibConfig.calibration_board_size
SQUARE_SIZE = CameraIntrinsicCalibConfig.square_size
MIN_IMAGES = CameraIntrinsicCalibConfig.min_calibration_images

# Get current camera info
ctx = rs.context()
if len(ctx.devices) == 1:
    d = ctx.devices[0]
    lgr.info(f"Found device: {d.get_info(rs.camera_info.name)}")
    CAMERA_SERIAL_NUMBER = d.get_info(rs.camera_info.serial_number)
elif len(ctx.devices) > 1:
    lgr.error("Multiple Intel RealSense devices detected. Please ensure only one is connected.")
    exit()
else:
    lgr.error("No Intel RealSense device connected. Please connect a device and try again.")
    exit()

# Create save directories
SAVE_DIR = CameraIntrinsicCalibConfig.save_data_path / CAMERA_SERIAL_NUMBER
SAVE_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_DIR = SAVE_DIR / "detection_images"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

def detect_checkerboard(image):
    """Detect checkerboard corners in the image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CALIBRATION_BOARD_SIZE, None)
    
    if ret:
        # Refine corner positions
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return True, corners_refined
    return False, None

def main():
    # Get camera serial number
    ctx = rs.context()
    if len(ctx.devices) > 0:
        for d in ctx.devices:
            lgr.info(f"Found device: {d.get_info(rs.camera_info.name)} "
                    f"Serial Number: {d.get_info(rs.camera_info.serial_number)}")
    else:
        
        raise RuntimeError("No Intel RealSense device connected")
    
    # Initialize realsense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 
                         CameraIntrinsicCalibConfig.camera_resolution[0], 
                         CameraIntrinsicCalibConfig.camera_resolution[1], 
                         rs.format.bgr8, 
                         30)
    pipeline.start(config)
    
    lgr.info(f"Camera intrinsic calibration started. Target: More than {MIN_IMAGES} images")
    lgr.info(f"Checkerboard size: {CALIBRATION_BOARD_SIZE}, Square size: {SQUARE_SIZE}mm")
    
    # Prepare object points
    objp = np.zeros((CALIBRATION_BOARD_SIZE[0] * CALIBRATION_BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CALIBRATION_BOARD_SIZE[0], 0:CALIBRATION_BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    
    # Arrays to store data points
    all_imgpoints = []  # 2D points in image plane
    captured_images = []
    save_idx = 0
    data_saved = False

    try:
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
            
            # Draw corners if detected
            if detected:
                cv2.drawChessboardCorners(display_image, CALIBRATION_BOARD_SIZE, corners, detected)
                cv2.putText(display_image, f"Checkerboard detected - Press 's' to save #({save_idx})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_image, "No checkerboard detected", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show progress
            cv2.putText(display_image, f"Images captured: {save_idx} (min: {MIN_IMAGES})", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if save_idx >= MIN_IMAGES:
                cv2.putText(display_image, "Press 'c' to calibrate, 's' to save more, 'q' to quit", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow("Camera Calibration - Collect Images", display_image)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') and save_idx >= MIN_IMAGES:
                lgr.info("Exiting and saving calibration data...")
                data_saved = True
                break
            elif key == ord('s') and detected:
                # Save image and corner data
                img_path = SAVE_DIR / f"calib_{save_idx:03d}.png"
                cv2.imwrite(str(img_path), color_image)
                
                # Save detection visualization
                debug_path = DEBUG_DIR / f"detection_{save_idx:03d}.png"
                cv2.imwrite(str(debug_path), display_image)
                
                # Store calibration data
                all_imgpoints.append(corners)
                captured_images.append(str(img_path))
                
                lgr.success(f"Image {save_idx + 1} captured and saved")
                save_idx += 1
            elif key == ord('s') and not detected:
                lgr.warning("Checkerboard not detected - image not saved")

    finally:
        cv2.destroyAllWindows()
        pipeline.stop()
    
    # Save calibration data to file if requested
    if data_saved:
        data_to_save = {
            'objp': objp,
            'imgpoints': all_imgpoints,
            'image_paths': captured_images,
            'image_size': color_image.shape[:2][::-1]  # (width, height)
        }
        
        data_path = SAVE_DIR / "calibration_data.pkl"
        with open(data_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        lgr.success(f"✅ Calibration data saved to {data_path}")
    else:
        lgr.info("Calibration data not saved")

if __name__ == "__main__":
    main()