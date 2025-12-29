import cv2
import numpy as np
from pathlib import Path
import sys
from loguru import logger as lgr
import json
from scipy.spatial.transform import Rotation as R

import pyrealsense2 as rs
from xarm.wrapper import XArmAPI


# Import configuration
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

def load_hand_eye_calibration():
    """Load hand-eye calibration results from calibrated data"""
    from config.config import HandInEyeCalibConfig
    
    # Let user choose directory name under BASE_DIR
    lgr.info("Please choose a hand-eye calibration experiment:")
    base_dir = HandInEyeCalibConfig.save_data_path
    
    # List all directories under BASE_DIR
    if base_dir.exists():
        directories = [d.name for d in base_dir.iterdir() if d.is_dir()]
    else:
        directories = []
    
    if not directories:
        lgr.error(f"No hand-eye calibration directories found in {base_dir}")
        return None
    
    lgr.info("Available hand-eye calibration experiments:")
    for i, dir_name in enumerate(directories, 1):
        lgr.info(f"{i} - {dir_name}")
    
    selected_exp = None
    while True:
        try:
            choice = input("Enter the number of your choice: ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(directories):
                selected_exp = directories[choice_idx]
                lgr.info(f"Selected experiment: {selected_exp}")
                break
            else:
                lgr.warning(f"Invalid choice. Please enter a number between 1 and {len(directories)}")
        except ValueError:
            lgr.warning("Please enter a valid number")
        except KeyboardInterrupt:
            lgr.info("Operation cancelled by user")
            return None
    
    # Construct path to calibration results
    calib_dir = base_dir / selected_exp
    result_path = calib_dir / "calibration_results.json"
    
    if not result_path.exists():
        lgr.error(f"Hand-eye calibration results not found at {result_path}")
        lgr.info("Available files in selected directory:")
        if calib_dir.exists():
            for file in calib_dir.iterdir():
                lgr.info(f"  - {file.name}")
        return None
    
    try:
        with open(result_path, 'r') as f:
            results = json.load(f)
        
        # Find the method with lowest error
        best_method = min(results.keys(), key=lambda k: results[k]['error'])
        best_result = results[best_method]
        
        lgr.info(f"Loading hand-eye calibration from: {result_path}")
        lgr.info(f"Best method: {best_method} (Error: {best_result['error']:.4f} px)")
        
        # Convert from cam2gripper to cam2base transformation
        R_cam2gripper = np.array(best_result['R'])
        t_cam2gripper = np.array(best_result['t']).reshape(3, 1)
        
        # Create 4x4 transformation matrix (cam to gripper)
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = R_cam2gripper
        T_cam2gripper[:3, 3] = t_cam2gripper.flatten()
        
        lgr.info(f"Camera to gripper transformation matrix:\n{T_cam2gripper}")
        
        return T_cam2gripper
        
    except Exception as e:
        lgr.error(f"Failed to load hand-eye calibration: {e}")
        return None

# Load hand-eye calibration or use default
cam_to_gripper = load_hand_eye_calibration()
if cam_to_gripper is None:
    lgr.warning("Using default camera to base transformation matrix")
    # Fallback to your original hardcoded matrix
    cam_to_gripper = np.array([
        [0.012247158513993717, -0.999895928742641, -0.007624879817037132, 0.21856645975937857],
        [0.9997757035321371, 0.012113206937224397, 0.01737276156850833, -0.03851832430393068],
        [-0.017278591816272865, -0.00783593654818163, 0.9998200079830667, -0.4060025918296039],
        [0, 0, 0, 1]
    ], dtype=np.float32)

class CalibrationValidator:
    def __init__(self, robot_ip):
        self.arm = None
        self.pipeline = None
        self.align = None
        self.color_intrinsics = None
        
        self.force_sensor_offset = 0  # 6.5cm offset in mm
        
        self.initialize_camera()
        self.initialize_robot(robot_ip)
        self.current_position = self.arm.get_position()[1] # Initial pose (x, y, z, roll, pitch, yaw)
        
        # Detection state
        self.frozen_frame = None
        self.input_point = None

    def initialize_robot(self, ip):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(True)
        self.arm.set_mode(0)
        self.arm.set_state(0)

    def initialize_camera(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        color_profile = profile.get_stream(rs.stream.color)
        self.color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        lgr.info(f"Camera on-chip intrinsics: {self.color_intrinsics}")
        # Ask user to confirm loading intrinsics, otherwise do not change them
        lgr.info("Press 'y' to load manually calibrated intrinsics")
        while True:
            key = input("(y/n): ").strip().lower()
            if key == 'y':
                if self.load_manual_intrinsic():
                    break
                else:
                    lgr.error("Failed to load manual intrinsics. Using default camera intrinsics.")
                    break
            elif key == 'n':
                lgr.info("Using default camera intrinsics.")
                break
            else:
                lgr.warning("Invalid input. Please enter 'y' or 'n'.")
        self.align = rs.align(rs.stream.color)

    # Camera intrinsics (load from calibrated data)
    def load_manual_intrinsic(self):
        """Load camera intrinsics from calibrated data and update self.color_intrinsics"""
        from config.config import CameraIntrinsicCalibConfig
        
        # Get camera serial number
        ctx = rs.context()
        camera_serial = None
        if len(ctx.devices) == 1:
            camera_serial = ctx.devices[0].get_info(rs.camera_info.serial_number)
            camera_name = ctx.devices[0].get_info(rs.camera_info.name)
            
            # Give user option to use live camera intrinsics or calibrated data
            lgr.info(f"Found connected camera: {camera_name} (Serial: {camera_serial})")
            
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
                            return False
                else:
                    lgr.error("No camera calibrations found")
                    return False
            else:
                lgr.error("Camera calibration directory not found")
                return False
        
        calib_dir = CameraIntrinsicCalibConfig.save_data_path / camera_serial
        
        # Try to load from .npz file first (most efficient)
        npz_path = calib_dir / "camera_params.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            camera_matrix = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            
            # Update self.color_intrinsics with loaded calibration data
            self.color_intrinsics.fx = float(camera_matrix[0, 0])
            self.color_intrinsics.fy = float(camera_matrix[1, 1])
            self.color_intrinsics.ppx = float(camera_matrix[0, 2])
            self.color_intrinsics.ppy = float(camera_matrix[1, 2])
            
            # Flatten the dist_coeffs array and convert to list of floats
            dist_coeffs_flat = dist_coeffs.flatten()
            self.color_intrinsics.coeffs = [float(dist_coeffs_flat[i]) for i in range(min(5, len(dist_coeffs_flat)))]
            
            lgr.info(f"✅ Loaded and applied camera intrinsics from {npz_path}")
            lgr.info(f"Camera intrinsics: {self.color_intrinsics}")
            return True

        # Fallback to JSON file
        json_path = calib_dir / "camera_calibration.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                calib_data = json.load(f)
            camera_matrix = np.array(calib_data['camera_matrix'])
            dist_coeffs = np.array(calib_data['distortion_coefficients'])
            
            # Update self.color_intrinsics with loaded calibration data
            self.color_intrinsics.fx = float(camera_matrix[0, 0])
            self.color_intrinsics.fy = float(camera_matrix[1, 1])
            self.color_intrinsics.ppx = float(camera_matrix[0, 2])
            self.color_intrinsics.ppy = float(camera_matrix[1, 2])
            
            # Flatten the dist_coeffs array and convert to list of floats
            dist_coeffs_flat = dist_coeffs.flatten()
            self.color_intrinsics.coeffs = [float(dist_coeffs_flat[i]) for i in range(min(5, len(dist_coeffs_flat)))]
            
            lgr.info(f"✅ Loaded and applied camera intrinsics from {json_path}")
            lgr.info(f"Camera intrinsics: {self.color_intrinsics}")
            return True
        
        lgr.error(f"❌ Camera calibration files not found in {calib_dir}")
        lgr.info("Available files:")
        if calib_dir.exists():
            for file in calib_dir.iterdir():
                lgr.info(f"  - {file.name}")
        return False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.frozen_frame is not None:
            self.input_point = (x, y)
            print(f"Selected point at ({x}, {y})")

    def xarm_translation_euler_2_matrix(self, translation_euler):
        '''
        **Input:**

        - pose: numpy array of shape (6,)

        **Output:**

        - Homogeneous transformation matrix of shape (4,4).
        '''
        # Convert Euler angles (roll, pitch, yaw) to rotation matrix
        pose_euler = translation_euler.copy()
        x, y, z, roll, pitch, yaw = translation_euler
        r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)
        rotation_matrix = r.as_matrix()

        # Create a 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix

        # Set the translation part
        T[0, 3] = x / 1000  # Convert mm to meters
        T[1, 3] = y / 1000  # Convert mm to meters
        T[2, 3] = z / 1000  # Convert mm to meters

        return T

    def transform_to_base(self, point_cam):
        point_cam = np.append(point_cam, 1)
        point_base = cam_to_gripper @ point_cam
        pose = self.arm.get_position()[1]
        pose = self.xarm_translation_euler_2_matrix(pose)
        point_base = np.dot(pose, point_base)
        return point_base[:3] * 1000  # Convert to mm

    def get_point_position(self, depth_frame):
        if self.input_point is None:
            return None

        x, y = self.input_point
        
        # Get depth at the clicked point
        depth_value = depth_frame.get_distance(x, y)
        if depth_value == 0:
            print("No depth data at selected point")
            return None

        # Convert pixel coordinates to 3D point in camera frame
        point_3d = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [x, y], depth_value
        )

        # Transform to base frame (mm)
        point_base = self.transform_to_base(np.array(point_3d))

        print(f"3D Point in camera frame: {point_3d}")
        print(f"3D Point in base frame: {point_base}")

        return point_base

    def move_to_position(self, position):
        target_pose = [
            position[0],
            position[1],
            position[2] + self.force_sensor_offset,
            self.current_position[3],  # Maintain roll
            self.current_position[4],  # Maintain pitch
            self.current_position[5]   # Maintain yaw
        ]
        self.arm.set_position(*target_pose, speed=40, wait=True)
        return self.arm.get_position()

    def run_validation(self):
        cv2.namedWindow('Camera Feed')
        cv2.setMouseCallback('Camera Feed', self.mouse_callback)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if not color_frame or not depth_frame:
                    continue

                # Use current frame or frozen frame for display
                if self.frozen_frame is None:
                    color_image = np.asanyarray(color_frame.get_data())
                    display_image = color_image.copy()
                else:
                    display_image = self.frozen_frame.copy()

                # Draw selection state
                if self.input_point is not None:
                    cv2.circle(display_image, self.input_point, 5, (0, 255, 0), -1)
                    cv2.putText(display_image, "Target", (self.input_point[0]+10, self.input_point[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow('Camera Feed', display_image)

                key = cv2.waitKey(1)
                if key == ord('f'):
                    self.frozen_frame = np.asanyarray(color_frame.get_data())
                    print("Frame frozen - click a point")
                elif key == ord('s') and self.input_point is not None:
                    point_position = self.get_point_position(depth_frame)
                    if point_position is not None:
                        print(f"\nTarget position: {point_position} mm")

                        # Move to the point
                        actual_pos = self.move_to_position(point_position)
                        print(f"Robot moved to position: {actual_pos}")
                elif key == ord('r'):
                    self.frozen_frame = None
                    self.input_point = None
                    print("Selection reset")
                elif key in (ord('q'), 27):
                    break

        finally:
            self.pipeline.stop()
            self.arm.disconnect()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    validator = CalibrationValidator('192.168.1.204')
    validator.run_validation()