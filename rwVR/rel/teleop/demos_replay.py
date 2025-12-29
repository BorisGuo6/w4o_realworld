import time 
import viser

import numpy as np 
from scipy.spatial.transform import Rotation as R
from rel import DATA_PATH

from rel.robots.pk_robot import XArm6WOEE, XArm7WOEE
from rel.utils import as_mesh


def replay_teleop_data(server: viser.ViserServer, exp_dir, replay_idx, arm):
    teleop_data = np.load(exp_dir / f"{replay_idx}.npz", allow_pickle=True)
    pointclouds = teleop_data["pointclouds"]
    proprioceptions = teleop_data["proprioceptions"]
    original_actions = teleop_data["original_actions"]
    demo_length = len(pointclouds)

    def update_scene(i):
        pointcloud = pointclouds[i]
        proprioception = proprioceptions[i]  # (8,)
        original_action = original_actions[i]  # (13,)
        
        action_wxyz = R.from_matrix(original_action[3:12].reshape(3, 3)).as_quat()[[3, 0, 1, 2]]
        action_pos = original_action[:3]  
    
    
        server.scene.add_point_cloud(
            f"pointcloud",
            pointcloud,
            colors=(255, 0, 0),
            point_size=0.005,
            point_shape="circle",
        )
        current_arm_mesh = arm.get_state_trimesh(proprioception[:-1], visual=True, collision=False)["visual"]
        server.scene.add_mesh_trimesh("current_arm_mesh", current_arm_mesh)
        server.scene.add_frame(
            position=action_pos,
            wxyz=action_wxyz,
            name="action",
        )
        
        
    timesteps_slider = server.gui.add_slider(
        "timesteps",
        min=0,
        max=demo_length - 1,
        initial_value=0,
        step=1,
    )
        
    timesteps_slider.on_update(lambda _: update_scene(timesteps_slider.value))
    update_scene(0)
        
        

if __name__ == "__main__":
    
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    # 0. set your hyperparameters
    robot_type = "xarm7"
    exp_dir = DATA_PATH / "0427_9points" / "raw_data"
    replay_idx = 4
    
            
    # 4. set your robot type: XArm6WOEE or XArm7WOEE
    robot_type = 'XArm7WOEE'
    assert robot_type in ['XArm6WOEE', 'XArm7WOEE'], "robot_type must be either XArm6WOEE or XArm7WOEE"
    if robot_type == 'XArm6WOEE':
        arm = XArm6WOEE()
    elif robot_type == 'XArm7WOEE':
        arm = XArm7WOEE()
    
    
    server = viser.ViserServer()
    replay_teleop_data(server, exp_dir, replay_idx, arm)
    
    while True:
        time.sleep(0.01)