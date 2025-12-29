import viser 
from rel import CAMERA_DATA_PATH
import cv2
import time
import torch
import viser 
import numpy as np 
from tqdm import tqdm
from loguru import logger as lgr
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE
from rel.cameras.realsense import Realsense
from rel.cameras.orbbec import Orbbec
from rel.cameras.nvdiffrast_renderer import NVDiffrastRenderer
from rel.utils import as_mesh, robust_compute_rotation_matrix_from_ortho6d


if __name__ == '__main__':
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = 'CL8H74100BB'
    exp_name = "0427_excalib_capture00"
    robot_type = "xarm7"
    camera_type = "orbbec"
    
    # 1. setup camera
    if camera_type == "orbbec":
        camera = Orbbec(serial_number, use_color=False, use_depth=True)
    elif camera_type == "realsense":
        camera = Realsense(serial_number)
    else:
        raise ValueError
    
    H, W = camera.h, camera.w
    K = camera.K
    
    # 3. load camera extrinsic params
    X_BaseCamera_path = CAMERA_DATA_PATH / serial_number/ exp_name / "manual2_X_BaseCamera.npy"
    X_BaseCamera = np.load(X_BaseCamera_path)
    X_CameraBase = np.linalg.inv(X_BaseCamera)
    
    

    ##########################################################################################
    # Main code
    ##########################################################################################
    server = viser.ViserServer()
    
    minx_handle = server.gui.add_slider("minx", -1, 3, 0.01, 0.4)
    maxx_handle = server.gui.add_slider("maxx", -1, 3, 0.01, 0.8)
    miny_handle = server.gui.add_slider("miny", -2, 2, 0.01, -0.3)
    maxy_handle = server.gui.add_slider("maxy", -2, 2, 0.01, 0.3)
    minz_handle = server.gui.add_slider("minz", -1, 3, 0.01, 0.03)
    maxz_handle = server.gui.add_slider("maxz", -1, 3, 0.01, 2) 
    
    while True:
        # get the camera pc 
        rt_data = camera.getCurrentData(pointcloud=True)
        pc = rt_data['pointcloud_np']
        pc = (X_BaseCamera[:3, :3] @ pc.T + X_BaseCamera[:3, 3][:, np.newaxis]).T
        
        minx_handle.value = min(minx_handle.value, maxx_handle.value)
        miny_handle.value = min(miny_handle.value, maxy_handle.value)
        minz_handle.value = min(minz_handle.value, maxz_handle.value)
        cropped_pc = pc[
            (pc[:, 0] > minx_handle.value) & (pc[:, 0] < maxx_handle.value) &
            (pc[:, 1] > miny_handle.value) & (pc[:, 1] < maxy_handle.value) &
            (pc[:, 2] > minz_handle.value) & (pc[:, 2] < maxz_handle.value)
        ]
        
        server.scene.add_point_cloud(
            "pc",
            pc,
            colors=(255, 0, 0),
            point_size=0.0015,
            point_shape="circle",
        )
        server.scene.add_point_cloud(
            "cropped_pc",
            cropped_pc,
            colors=(0, 255, 0),
            point_size=0.0015,
            point_shape="circle",
        )
        
        
        time.sleep(0.01)
        