import viser 
import zarr
import numpy as np
import time
# /
#  ├── data
#  │   ├── actions
#  │   │   ├── original_actions (2903, 13) float32
#  │   │   └── target_robot_points (2903, 3, 400) float32
#  │   └── obs
#  │       ├── current_robot_points (2903, 3, 400) float32
#  │       ├── point_clouds (2903, 3, 512) float32
#  │       └── robot0_all_qpos (2903, 8) float32
#  └── meta
#      ├── camera_meta
#      └── episode_ends (9,) int64

if __name__ == "__main__":
    data_path = "/home/rpt/rpt/rwVR/data/0428_9points_zixuan/9_17_rpt.zarr"
    
    # Load the zarr dataset
    dataset = zarr.open(data_path, mode='r')
    
    # Get the number of timesteps
    num_timesteps = dataset['data/obs/current_robot_points'].shape[0]
    
    # Create viser server
    server = viser.ViserServer()
    
    # Add a slider for timestep control
    timestep_handle = server.gui.add_slider(
        "Timestep",
        min=0,
        max=num_timesteps-1,
        step=1,
        initial_value=0
    )
    
    def update_visualization(timestep):
        # Get data for current timestep
        current_robot_points = dataset['data/obs/current_robot_points'][timestep]  # (3, 400)
        target_robot_points = dataset['data/actions/target_robot_points'][timestep]  # (3, 400)
        point_cloud = dataset['data/obs/point_clouds'][timestep]  # (3, 512)
        
        # Add current robot points (red)
        server.scene.add_point_cloud(
            "current_robot_points",
            current_robot_points.T,  # Transpose to (400, 3)
            colors=(255, 0, 0),  # Red
            point_size=0.005,
            point_shape="circle",
        )
        
        # Add target robot points (green)
        server.scene.add_point_cloud(
            "target_robot_points",
            target_robot_points.T,  # Transpose to (400, 3)
            colors=(0, 255, 0),  # Green
            point_size=0.005,
            point_shape="circle",
        )
        
        # Add point cloud (blue)
        server.scene.add_point_cloud(
            "point_cloud",
            point_cloud.T,  # Transpose to (512, 3)
            colors=(0, 0, 255),  # Blue
            point_size=0.003,
            point_shape="circle",
        )
    
    # Initial visualization
    update_visualization(0)
    
    timestep_handle.on_update(
        lambda _: update_visualization(timestep_handle.value)
    )
    # Keep the server running
    while True:
        time.sleep(1)