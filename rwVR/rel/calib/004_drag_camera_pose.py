import viser 
from rel import CAMERA_DATA_PATH
import time
import viser 
import numpy as np 
import trimesh 
from scipy.spatial.transform import Rotation as R
from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE, XArm7
from rel.robots.rw_robot import XArm6RealWorld, XArm7RealWorld
from rel.cameras.realsense import Realsense
try:
    from rel.cameras.orbbec import Orbbec
except:
    Orbbec = None
from rel.utils import as_mesh


if __name__ == '__main__':
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    serial_number = 'CL8H74100BB'
    exp_name = "0515_excalib_capture00"
    robot_type = 'XArm7WOEE'
    camera_type = "orbbec"

    # 5. set the input (Optional) & output path, by default it's 
    X_BaseCamera = np.eye(4)
    X_BaseCamera_manual_dir = CAMERA_DATA_PATH / serial_number / exp_name
    X_BaseCamera_manual_dir.mkdir(parents=True, exist_ok=True)
    X_BaseCamera_manual_path = X_BaseCamera_manual_dir / "manual_X_BaseCamera.npy"
    if X_BaseCamera_manual_path.exists():
        X_BaseCamera = np.load(X_BaseCamera_manual_path)
    
    ##########################################################################################
    # Main code
    ##########################################################################################
        
    # 1. setup camera
    if camera_type == "orbbec":
        camera = Orbbec(serial_number, use_color=True, use_depth=True)
    elif camera_type == "realsense":
        camera = Realsense(serial_number)
    else:
        raise ValueError
    
    H, W = camera.h, camera.w
    K = camera.K

    # 2. set your robot type: XArm6WOEE or XArm7WOEE
    assert robot_type in ['XArm6WOEE', 'XArm7WOEE', 'XArm7'], "robot_type must be either XArm6WOEE or XArm7WOEE or XArm7"
    if robot_type == 'XArm6WOEE':
        arm = XArm6WOEE()
        xarm_rw = XArm6RealWorld()
    elif robot_type == 'XArm7WOEE':
        arm = XArm7WOEE()
        xarm_rw = XArm7RealWorld()
    elif robot_type == 'XArm7':
        arm = XArm7()
        xarm_rw = XArm7RealWorld()


    server = viser.ViserServer()
    
    camera_wxyz = R.from_matrix(X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]]
    camera_pos = X_BaseCamera[:3, 3]
    camera_control = server.scene.add_transform_controls(
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
        rt_data = camera.getCurrentData(pointcloud=True)
        pc = rt_data['pointcloud_np']
        if camera_type == "orbbec":
            point_colors = (220, 70, 70)
        elif camera_type == "realsense":
            point_colors = np.asarray(rt_data['pointcloud_o3d'].colors)
            
        pc = (X_BaseCamera[:3, :3] @ pc.T + X_BaseCamera[:3, 3][:, np.newaxis]).T
        server.scene.add_point_cloud(
            "pc",
            pc,
            colors=point_colors,
            point_size=0.0005,
            point_shape="circle",
        )
        current_joint_values = xarm_rw.get_joint_values()
        if robot_type == 'XArm7':
            current_joint_values = np.concatenate([current_joint_values, [0.0] * 6])
        current_arm_mesh = arm.get_state_trimesh(current_joint_values, visual=True, collision=False)["visual"]
        current_arm_pc = trimesh.sample.sample_surface(as_mesh(current_arm_mesh), 20000)[0]
        
        server.scene.add_mesh_simple("current_arm_mesh", 
                                        as_mesh(current_arm_mesh).vertices,
                                        as_mesh(current_arm_mesh).faces,
                                        color=(62, 138, 252),
                                        opacity=0.2,
                                        )
        server.scene.add_point_cloud(
            "current_arm_pc",
            current_arm_pc,
            colors=(62, 138, 252),
            point_size=0.0015,
            point_shape="circle",
        )

    update_camera_pose(camera_control.wxyz, camera_control.position)
    
    save_button = server.gui.add_button("save_camera_pose")
    save_button.on_click(
        lambda _: np.save(X_BaseCamera_manual_path, X_BaseCamera)
    )
    

    while True:
        update_camera_pose(camera_control.wxyz, camera_control.position)
        time.sleep(0.1)
        
