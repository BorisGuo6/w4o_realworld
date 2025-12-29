import viser 
import trimesh
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from rel import SAM_PATH, SAM_TYPE
from rel.utils.foundationpose import single_frame_registration
from rel.cameras.realsense import Realsense
from rel.utils.sam_prompt_drawer import SAMPromptDrawer


OBJECT_MESH_ROOT_DIR_PATH = Path("/home/xzx/projects/dro_real/data/object_mesh")


def main(object_name, realsense_serial_number="317222073552"):

    mesh = trimesh.load(OBJECT_MESH_ROOT_DIR_PATH / object_name / f"{object_name}.obj")

    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    
    camera = Realsense(realsense_serial_number)
    
    for i in range(20):
        rtr_dict = camera.getCurrentData()
    
    rgb = rtr_dict["rgb"]
    depth = rtr_dict["depth"]
    pc_o3d = rtr_dict["pointcloud_o3d"]
    
    prompt_drawer.reset()
    mask = prompt_drawer.run(rgb)  # (720, 1280)
    
    X_CamObj = single_frame_registration(mesh, camera.K, rgb, depth, mask)

    server = viser.ViserServer()
    server.scene.add_point_cloud(
        "scene_pc",
        points=np.asarray(pc_o3d.points),
        colors=np.asarray(pc_o3d.colors),
        point_size=0.003,
    )
    server.scene.add_mesh_trimesh(
        "object_mesh",
        mesh,
        position=X_CamObj[:3, 3],
        wxyz=R.from_matrix(X_CamObj[:3, :3]).as_quat()[[3, 0, 1, 2]],  # xyzw -> wxyz
    )
    breakpoint()
    return X_CamObj


if __name__ == "__main__":
    main("dinosaur")
