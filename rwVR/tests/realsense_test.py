import time
import viser 
import numpy as np
from rel.cameras.realsense import Realsense


def main(realsense_serial_number="317222073552"):
    camera = Realsense(realsense_serial_number)
    server = viser.ViserServer()

    while True:
        rtr_dict = camera.getCurrentData()
        pc_o3d = rtr_dict["pointcloud_o3d"]
        
        server.scene.add_point_cloud(
            "scene_pc",
            points=np.asarray(pc_o3d.points),
            colors=np.asarray(pc_o3d.colors),
            point_size=0.003,
        )
        time.sleep(0.1)




if __name__ == "__main__":
    main()