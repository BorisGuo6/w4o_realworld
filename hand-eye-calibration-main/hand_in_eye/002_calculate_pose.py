import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger as lgr
import sys
import pyrealsense2 as rs

# Import configuration
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from config.config import HandInEyeCalibConfig

# Set parameters from config
CHECKERBOARD = HandInEyeCalibConfig.calibration_board_size
SQUARE_SIZE = HandInEyeCalibConfig.square_size
SAVE_DIR = HandInEyeCalibConfig.save_data_path / HandInEyeCalibConfig.exp_name
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Create debug directory for projection images
DEBUG_DIR = SAVE_DIR / "projections"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# Camera intrinsics (load from calibrated data)
def load_camera_intrinsics():
    """Load camera intrinsics from calibrated data or directly from connected camera"""
    from config.config import CameraIntrinsicCalibConfig
    
    # Get camera serial number or let user choose
    ctx = rs.context()
    if len(ctx.devices) == 1:
        camera_serial = ctx.devices[0].get_info(rs.camera_info.serial_number)
        camera_name = ctx.devices[0].get_info(rs.camera_info.name)
        
        # Give user option to use live camera intrinsics or calibrated data
        lgr.info(f"Found connected camera: {camera_name} (Serial: {camera_serial})")
        lgr.info("Choose intrinsic source:")
        lgr.info("1 - Use factory intrinsics from connected camera")
        lgr.info("2 - Load calibrated intrinsics from file")
        
        while True:
            try:
                choice = input("Enter your choice (1 or 2): ").strip()
                if choice == "1":
                    # Load intrinsics directly from camera
                    pipeline = rs.pipeline()
                    config = rs.config()
                    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                    
                    try:
                        profile = pipeline.start(config)
                        color_stream = profile.get_stream(rs.stream.color)
                        color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
                        
                        camera_matrix = np.array([[color_intrinsics.fx, 0, color_intrinsics.ppx],
                                                [0, color_intrinsics.fy, color_intrinsics.ppy],
                                                [0, 0, 1]])
                        
                        # RealSense uses Brown-Conrady distortion model
                        dist_coeffs = np.array([[color_intrinsics.coeffs[0],  # k1
                                               color_intrinsics.coeffs[1],   # k2
                                               color_intrinsics.coeffs[2],   # p1
                                               color_intrinsics.coeffs[3],   # p2
                                               color_intrinsics.coeffs[4]]]) # k3
                        
                        pipeline.stop()
                        lgr.info("✅ Loaded factory intrinsics from connected camera")
                        lgr.info(f"Image size: {color_intrinsics.width} x {color_intrinsics.height}")
                        return camera_matrix, dist_coeffs
                        
                    except Exception as e:
                        lgr.error(f"Failed to get intrinsics from camera: {e}")
                        pipeline.stop()
                        return None, None
                        
                elif choice == "2":
                    break
                else:
                    lgr.warning("Please enter 1 or 2")
            except KeyboardInterrupt:
                lgr.info("Operation cancelled by user")
                return None, None
    else:
        camera_serial = None
        
    # Load from calibrated data
    if camera_serial is None:
        # Let user choose from available calibration directories
        base_dir = CameraIntrinsicCalibConfig.save_data_path
        if base_dir.exists():
            directories = [d.name for d in base_dir.iterdir() if d.is_dir()]
            if directories:
                lgr.info("Available camera calibrations:")
                for i, dir_name in enumerate(directories, 1):
                    lgr.info(f"{i} - {dir_name}")
                while True:
                    try:
                        choice = input("Select camera calibration: ").strip()
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(directories):
                            camera_serial = directories[choice_idx]
                            break
                        else:
                            lgr.warning(f"Invalid choice. Enter 1-{len(directories)}")
                    except (ValueError, KeyboardInterrupt):
                        lgr.error("Invalid input or cancelled")
                        return None, None
            else:
                lgr.error("No camera calibrations found")
                return None, None
        else:
            lgr.error("Camera calibration directory not found")
            return None, None
    
    calib_dir = CameraIntrinsicCalibConfig.save_data_path / camera_serial
    
    # Try to load from .npz file first (most efficient)
    npz_path = calib_dir / "camera_params.npz"
    if npz_path.exists():
        data = np.load(npz_path)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['dist_coeffs']
        lgr.info(f"✅ Loaded camera intrinsics from {npz_path}")
        return camera_matrix, dist_coeffs
    
    # Fallback to JSON file
    json_path = calib_dir / "camera_calibration.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            calib_data = json.load(f)
        camera_matrix = np.array(calib_data['camera_matrix'])
        dist_coeffs = np.array(calib_data['distortion_coefficients'])
        lgr.info(f"✅ Loaded camera intrinsics from {json_path}")
        return camera_matrix, dist_coeffs
    
    lgr.error(f"❌ Camera calibration files not found in {calib_dir}")
    lgr.info("Available files:")
    if calib_dir.exists():
        for file in calib_dir.iterdir():
            lgr.info(f"  - {file.name}")
    return None, None

# Load camera intrinsics
camera_matrix, dist_coeffs = load_camera_intrinsics()
if camera_matrix is None or dist_coeffs is None:
    lgr.error("Failed to load camera intrinsics. Exiting...")
    sys.exit(1)

lgr.info(f"Camera Matrix:\n{camera_matrix}")
lgr.info(f"Distortion Coefficients:\n{dist_coeffs.flatten()}")

def visualize_calibration_results(results):
    """Visualize the calibration results with error comparison"""
    methods = list(results.keys())
    errors = [results[m][0] for m in methods]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, errors)
    plt.title('Hand-Eye Calibration Methods Comparison')
    plt.ylabel('Average Reprojection Error (px)')
    plt.xlabel('Calibration Method')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate 3D checkerboard points
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Load metadata
    metadata_path = SAVE_DIR / "metadata.json"
    if not metadata_path.exists():
        lgr.error(f"Metadata file not found at {metadata_path}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    all_rvecs, all_tvecs = [], []
    R_gripper2base, t_gripper2base = [], []
    image_points_list, camera_positions, image_paths = [], [], []

    lgr.info(f"Processing {len(metadata)} data points...")

    for idx, entry in enumerate(metadata):
        image_path = entry['image_path']
        position = np.array(entry['position'], dtype=np.float64)
        orientation = np.array(entry['orientation'], dtype=np.float64)
        corners = np.array(entry['corners'], dtype=np.float32)

        image = cv2.imread(image_path)
        if image is None:
            lgr.warning(f"Could not load image: {image_path}")
            continue

        if corners.size == 0:
            lgr.warning(f"No corners found in image: {image_path}")
            continue

        if corners.ndim == 2:
            corners = corners.reshape(-1, 1, 2)
        image_points_list.append(corners.astype(np.float32))
        image_paths.append(image_path)

        ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        if not ret:
            lgr.warning(f"Failed to solve PnP for image: {image_path}")
            continue

        all_rvecs.append(rvec)
        all_tvecs.append(tvec)
        camera_positions.append(tvec.flatten())

        rx, ry, rz = np.deg2rad(orientation)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(rx), -np.sin(rx)],
                       [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                       [0, 1, 0],
                       [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                       [np.sin(rz), np.cos(rz), 0],
                       [0, 0, 1]])
        R = Rz @ Ry @ Rx
        t = position.reshape(3, 1)

        R_gripper2base.append(R)
        t_gripper2base.append(t)

    camera_positions = np.array(camera_positions)

    methods = {
        "Tsai": cv2.CALIB_HAND_EYE_TSAI,
        "Park": cv2.CALIB_HAND_EYE_PARK,
        "Horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "Andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "Daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS
    }

    results = {}
    for name, method in methods.items():
        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                [cv2.Rodrigues(rvec)[0] for rvec in all_rvecs],
                all_tvecs,
                method=method
            )

            total_error = 0
            valid_points = 0

            # Fix board pose in base frame using first pose
            R_cam2board_0, _ = cv2.Rodrigues(all_rvecs[0])
            t_cam2board_0 = all_tvecs[0]
            
            # Transform board pose from camera to gripper frame
            R_gripper2board_0 = R_cam2gripper @ R_cam2board_0
            t_gripper2board_0 = R_cam2gripper @ t_cam2board_0 + t_cam2gripper
            
            # Transform board pose from gripper to base frame
            R_base2board_0 = R_gripper2base[0] @ R_gripper2board_0
            t_base2board_0 = R_gripper2base[0] @ t_gripper2board_0 + t_gripper2base[0]

            for i in range(len(all_rvecs)):
                # Transform board from base to current gripper position
                R_gripper2base_inv = R_gripper2base[i].T
                t_gripper2base_inv = -R_gripper2base_inv @ t_gripper2base[i]
                
                R_gripper2board_i = R_gripper2base_inv @ R_base2board_0
                t_gripper2board_i = R_gripper2base_inv @ (t_base2board_0 - t_gripper2base[i])
                
                # Transform board from gripper to camera
                R_cam2gripper_inv = R_cam2gripper.T
                t_cam2gripper_inv = -R_cam2gripper_inv @ t_cam2gripper
                
                R_cam2board_expected = R_cam2gripper_inv @ R_gripper2board_i
                t_cam2board_expected = R_cam2gripper_inv @ (t_gripper2board_i - t_cam2gripper)
                
                rvec_expected, _ = cv2.Rodrigues(R_cam2board_expected)

                projected_points, _ = cv2.projectPoints(
                    objp, rvec_expected, t_cam2board_expected, camera_matrix, dist_coeffs
                )

                error = cv2.norm(image_points_list[i], projected_points, cv2.NORM_L2) / len(projected_points)
                total_error += error
                valid_points += 1

                if i < 3:
                    debug_img = cv2.imread(image_paths[i])
                    if debug_img is None:
                        continue
                    if len(debug_img.shape) == 2 or debug_img.shape[2] == 1:
                        debug_img = cv2.cvtColor(debug_img, cv2.COLOR_GRAY2BGR)

                    # Draw detected corners in red
                    for pt in image_points_list[i].reshape(-1, 2):
                        cv2.circle(debug_img, tuple(pt.astype(int)), 5, (0, 0, 255), -1)
                    # Draw reprojected corners in green
                    for pt in projected_points.reshape(-1, 2):
                        cv2.circle(debug_img, tuple(pt.astype(int)), 3, (0, 255, 0), 2)

                    cv2.putText(debug_img, f"{name} Error: {error:.4f} px", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    debug_filename = DEBUG_DIR / f"{name}_proj_{i}.png"
                    cv2.imwrite(str(debug_filename), debug_img)

            avg_error = total_error / valid_points if valid_points > 0 else float('inf')
            results[name] = (avg_error, R_cam2gripper, t_cam2gripper)

        except Exception as e:
            lgr.error(f"Error with {name} method: {str(e)}")
            continue

    lgr.info("\n📊 Hand-Eye Calibration Results (sorted by reprojection error):")
    for name, (err, R, t) in sorted(results.items(), key=lambda x: x[1][0]):
        lgr.info(f"{name:<10} Error: {err:.4f} px")
        lgr.info(f"Rotation matrix:\n{R}")
        lgr.info(f"Translation vector:\n{t.squeeze()}\n")

    visualize_calibration_results(results)

    result_path = SAVE_DIR / "calibration_results.json"
    with open(result_path, 'w') as f:
        json.dump({name: {"error": err, "R": R.tolist(), "t": t.squeeze().tolist()} 
                   for name, (err, R, t) in results.items()}, f, indent=4)
    lgr.info(f"Calibration results saved to {result_path}")

if __name__ == "__main__":
    main()