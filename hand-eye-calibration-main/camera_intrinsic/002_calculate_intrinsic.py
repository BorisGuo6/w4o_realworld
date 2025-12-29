import numpy as np
import cv2
import json
import os
import sys
import pickle
from pathlib import Path
import pyrealsense2 as rs
from loguru import logger as lgr
import matplotlib.pyplot as plt

# Add parent directory to path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import configuration
from config.config import CameraIntrinsicCalibConfig

# Configuration parameters
CALIBRATION_BOARD_SIZE = CameraIntrinsicCalibConfig.calibration_board_size
SQUARE_SIZE = CameraIntrinsicCalibConfig.square_size
BASE_DIR = CameraIntrinsicCalibConfig.save_data_path
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
    # Let user choose directory name under BASE_DIR
    lgr.info("Please choose a directory name for your saved calibration data:")
        # List all directories under BASE_DIR
    if BASE_DIR.exists():
        directories = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
    else:
        directories = []
    
    if directories:
        lgr.info("Available directories:")
        for i, dir_name in enumerate(directories, 1):
            lgr.info(f"{i} - {dir_name}")
        
        while True:
            try:
                choice = input("Enter the number of your choice: ").strip()
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(directories):
                    CAMERA_SERIAL_NUMBER = directories[choice_idx]
                    lgr.info(f"Selected directory: {CAMERA_SERIAL_NUMBER}")
                    break
                else:
                    lgr.warning(f"Invalid choice. Please enter a number between 1 and {len(directories)}")
            except ValueError:
                lgr.warning("Please enter a valid number")
            except KeyboardInterrupt:
                lgr.info("Operation cancelled by user")
                exit()

SAVE_DIR = BASE_DIR / CAMERA_SERIAL_NUMBER

def visualize_calibration_results(camera_matrix, dist_coeffs, mean_error, per_image_errors, image_paths):
    """Visualize calibration results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot per-image reprojection errors
    ax1.bar(range(len(per_image_errors)), per_image_errors)
    ax1.axhline(y=mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.4f}')
    ax1.set_xlabel('Image Index')
    ax1.set_ylabel('Reprojection Error (pixels)')
    ax1.set_title('Per-Image Reprojection Errors')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Display camera parameters as text
    ax2.axis('off')
    params_text = f"""Camera Intrinsic Parameters:

Focal Length (fx): {camera_matrix[0,0]:.2f} px
Focal Length (fy): {camera_matrix[1,1]:.2f} px
Principal Point (cx): {camera_matrix[0,2]:.2f} px
Principal Point (cy): {camera_matrix[1,2]:.2f} px

Distortion Coefficients:
k1: {dist_coeffs[0,0]:.6f}
k2: {dist_coeffs[0,1]:.6f}
p1: {dist_coeffs[0,2]:.6f}
p2: {dist_coeffs[0,3]:.6f}
k3: {dist_coeffs[0,4]:.6f}

Mean Reprojection Error: {mean_error:.4f} px
Total Images Used: {len(per_image_errors)}"""
    
    ax2.text(0.1, 0.9, params_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "calibration_results.png", dpi=150, bbox_inches='tight')
    plt.show()

def main():
    # Load calibration data
    data_path = SAVE_DIR / "calibration_data.pkl"
    
    if not os.path.exists(data_path):
        lgr.error(f"Calibration data not found at {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        calibration_data = pickle.load(f)
    
    # Extract data
    objp = calibration_data['objp']
    imgpoints = calibration_data['imgpoints']
    image_paths = calibration_data['image_paths']
    img_shape = calibration_data['image_size']  # (width, height)
    
    num_images = len(imgpoints)
    lgr.info(f"Loaded calibration data with {num_images} images")
    
    # Check if we have enough images
    if num_images < MIN_IMAGES:
        lgr.warning(f"⚠️ Not enough images for calibration. Collected: {num_images}, Required: {MIN_IMAGES}")
        return
    
    # Prepare object points (same for all images)
    objpoints = [objp] * num_images
    
    lgr.info(f"Performing camera calibration with {num_images} images...")
    
    # Calibrate camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    
    if ret:
        # Calculate reprojection error
        total_error = 0
        per_image_errors = []
        
        for i in range(num_images):
            imgpoints_projected, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            per_image_errors.append(error)
            total_error += error
        
        mean_error = total_error / num_images
        
        # Save calibration results
        calibration_data = {
            "camera_matrix": camera_matrix.tolist(),
            "distortion_coefficients": dist_coeffs.tolist(),
            "image_size": img_shape,
            "mean_reprojection_error": mean_error,
            "per_image_errors": per_image_errors,
            "num_images_used": num_images,
            "checkerboard_size": (CALIBRATION_BOARD_SIZE[0], CALIBRATION_BOARD_SIZE[1]),
            "square_size": SQUARE_SIZE,
            "captured_images": image_paths
        }
        
        # Save to JSON
        with open(SAVE_DIR / "camera_calibration.json", 'w') as f:
            json.dump(calibration_data, f, indent=4)
        
        # Save camera matrix and distortion coefficients
        np.savez(SAVE_DIR / "camera_params.npz", 
                 camera_matrix=camera_matrix, 
                 dist_coeffs=dist_coeffs)
        
        lgr.success("✅ Camera calibration completed successfully!")
        lgr.info(f"📊 Mean reprojection error: {mean_error:.4f} pixels")
        lgr.info(f"📁 Results saved to: {SAVE_DIR}")
        
        # Visualize results
        visualize_calibration_results(camera_matrix, dist_coeffs, mean_error, per_image_errors, image_paths)
        
    else:
        lgr.error("❌ Camera calibration failed!")

if __name__ == "__main__":
    main()