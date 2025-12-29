import time
import torch
import viser 
import numpy as np 
from scipy.spatial.transform import Rotation as R
from rel import CAMERA_DATA_PATH
from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE
from rel.cameras.realsense import Realsense
from rel.cameras.nvdiffrast_renderer import NVDiffrastRenderer
from rel.utils import as_mesh



if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your exp_name
    exp_name = "0212_excalib_capture00"
    # 1. init your camera pose, you can just use the default one
    X_BaseCamera = np.eye(4)
    X_BaseCamera[0:3, 0:3] = R.from_euler('y', 180, degrees=True).as_matrix() @ R.from_euler('x', 55, degrees=True).as_matrix()
    X_BaseCamera[0:3, 3] = np.array([0.5, 0.3, 0.3])
    # 2. get realsense serial number as described in doc, and paste it here
    serial_number = "233622079809"
    camera = Realsense(serial_number)
    # 3. (Optional) set the camera intrinsics, you can just use the default one
    # K = np.load(K_path)
    # camera.set_intrinsics(K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    
    # 4. set your robot type: XArm6WOEE or XArm7WOEE
    robot_type = 'XArm7WOEE'
    assert robot_type in ['XArm6WOEE', 'XArm7WOEE'], "robot_type must be either XArm6WOEE or XArm7WOEE"
    if robot_type == 'XArm6WOEE':
        arm = XArm6WOEE()
    elif robot_type == 'XArm7WOEE':
        arm = XArm7WOEE()
        
    # 5. set the output path, by default it's 
    init_X_BaseCamera_dir = CAMERA_DATA_PATH / serial_number / exp_name
    init_X_BaseCamera_path = init_X_BaseCamera_dir / "init_X_BaseCamera.npy"
    init_X_BaseCamera_dir.mkdir(parents=True, exist_ok=True)
    
    ##########################################################################################
        
        
    ##########################################################################################
    # Main code 
    ##########################################################################################
    # setup virtual camera and robot mesh 
    H, W = camera.h, camera.w
    K = camera.K
    renderer = NVDiffrastRenderer([H, W])
    rw_ref_joint_values = arm.reference_joint_values_np
    rw_ref_arm_visual_mesh = as_mesh(arm.get_state_trimesh(rw_ref_joint_values)['visual'])
    
    # setup visualization and camera control 
    sv = viser.ViserServer()
    sv.scene.add_mesh_trimesh(
        name="robot_mesh",
        mesh=rw_ref_arm_visual_mesh,
    )
    camera_wxyz = R.from_matrix(X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]]
    camera_pos = X_BaseCamera[:3, 3]
    camera_control = sv.scene.add_transform_controls(
        f"camera_pose",
        opacity=0.75,
        disable_sliders=True,
        scale=0.25,
        line_width=2.5,
        wxyz=camera_wxyz,
        position=camera_pos,
    )

    # on update camera control, rerender the robot mesh
    def update_camera_pose(camera_wxyz, camera_pos):
        global X_BaseCamera
        X_BaseCamera = np.eye(4)
        X_BaseCamera[:3, :3] = R.from_quat(camera_wxyz[[1, 2, 3, 0]]).as_matrix()
        X_BaseCamera[:3, 3] = camera_pos
        X_CameraBase = np.linalg.inv(X_BaseCamera)
        mask = renderer.render_mask(torch.from_numpy(rw_ref_arm_visual_mesh.vertices).cuda().float(),
            torch.from_numpy(rw_ref_arm_visual_mesh.faces).cuda().int(),
            torch.from_numpy(K).cuda().float(),
            torch.from_numpy(X_CameraBase).cuda().float()
        ).detach().cpu().numpy()
        mask_as_img = np.stack([mask* 255]*3, axis=-1).astype(np.uint8)         
        sv.scene.add_camera_frustum(
            "nvidiff_mask", 
            fov=camera.fov_x, 
            aspect=camera.aspect_ratio, 
            wxyz=camera_wxyz, 
            position=camera_pos, 
            image=mask_as_img, 
            scale=0.2
        )
        
        # add the realworld rgb image as reference 
        rt_dict = camera.getCurrentData(pointcloud=False)
        rgb_image = rt_dict["rgb"]
        sv.scene.add_camera_frustum(
            "realsense_image", 
            fov=camera.fov_x, 
            aspect=camera.aspect_ratio, 
            wxyz=camera_wxyz, 
            position=camera_pos, 
            image=rgb_image, 
            scale=0.2
        )
        
    camera_control.on_update(
        lambda _: update_camera_pose(camera_control.wxyz, camera_control.position)
    )
    
    save_button = sv.gui.add_button("save_camera_pose",)
    save_button.on_click(
        lambda _: np.save(init_X_BaseCamera_path, X_BaseCamera)
    )
    
    while True: 
        time.sleep(1)
        
        