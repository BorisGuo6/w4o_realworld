# Hand-to-Eye Calibration for Robotic Manipulators

A calibration system for eye-to-hand configuration supporting XArm7 using Intel RealSense cameras and based on principles from [EasyHec](https://github.com/ootts/EasyHeC).

## Prerequisites
The repo is tested on Ubuntu 22.04, with Python 3.9. \
Nvidia CUDA 12.4 and PyTorch 2.6.0.

## Installation

Before cloning the repository, please contact Roboscience for installation permissions.

```bash
# Clone repository
git clone https://github.com/roboscience-ai/hand-to-eye-calibration.git
cd hand-to-eye-calibration

# Create conda environment (optional)
conda create -n calib python=3.9
conda activate calib

# Install dependencies
pip install -r requirements.txt

# Install Pytorch
# Please check the installation instructions (https://pytorch.org/) for your system
pip3 install torch torchvision torchaudio

# Install submodules
git submodule update --init --recursive

# Install SAM
cd thirdparty/sam
pip install -e .

# Install nvdiffrast
cd thirdparty/nvdiffrast
pip install .
```
For Python3.9, we need to manually fix mplib for compatibility, please refer to [this issue](https://github.com/haosulab/MPlib/issues/98).

## Usage
### Hand-to-Eye Calibration
1. Download SAM checkpoint (by default we use sam_vit_h_4b8939.pth, download [here](https://huggingface.co/HCMUE-Research/SAM-vit-h/blob/main/sam_vit_h_4b8939.pth)).
2. **Modify the customizable parameters of your calibration system in config/config.py, HandToEyeCalibConfig.**
3. `python hand_to_eye/001_init_camera_pose.py` \
    Note: Open local host url, drag the axis in gui until the robot arm mesh overlaps with the actual world robot arm. Click save_camera_pose to save this pose as initial camera pose.
4. `python hand_to_eye/002_capture_data.py` \
    Note: Move the robot arm around manually (or teleport) and collect around 30 different poses of the robot. Ensure the robot arm is visible inside the camera frame.
5. `python hand_to_eye/003_sam_robot_arms.py` \
    Note: Manually segment the robot arm in each frame using SAM. Press 's' to save mask, 'z' to retry drawing box, and 'q' to quit.
6. `python hand_to_eye/004_camera_pose_opt.py` \
    Note: The optimized camera pose data will be found in the save_dir in the config.
### Custom Robot Arm Hand-to-Eye Setup
1. Change the URDF path in config to your custom robot arm URDF.
2. Write your own robot arm class inside utils/arm_pk.py and utils/arm_rw.py, detailed instructions can be found inside the scripts.
3. After writing your own robot arm class, change the names of the custom class (e.g. robot_class_rw/robot_class_pk) to your own custom class name in config for import.

### Hand-in-Eye Calibration
1. **Modify the customizable parameters of your calibration system in config/config.py, HandInEyeCalibConfig.**
2.  `python hand_in_eye/001_capture_data.py ip=1.1.1.1` or \
    `python hand_in_eye/001_capture_data.py` (directly use ip from config) \
    Note: Fix the calibration chessboard, move the robot arm around manually (or teleport) and collect around 40 different poses of the robot. **Press 's' to save current data and 'q' to quit.** Ensure the calibration chessboard is in sight, otherwise the system would not allow saving the data point.
3. `python hand_in_eye/002_calculate_pose.py` \
    Note: After running, you can see the visualization of corner point reprojection under save_dir/projections. Usually, the one with lowest error have the best calibration outcome.
4. After running all scripts, you can find your optimized camera pose under end effector frame inside the save_dir in your config.
