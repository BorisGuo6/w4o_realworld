from rel.cameras.orbbec import Orbbec
from model import GroundedGraspNet


## camera
serial_number = "CL8H74100BB"
camera = Orbbec(serial_number=serial_number, use_depth=True, use_color=True)

obs = camera.getCurrentData(pointcloud=True)

print(obs.keys())
exit()

## model
ckpt_path = f"ckpt/checkpoint-rs.tar"
grounded_graspnet = GroundedGraspNet(ckpt_path=ckpt_path)

prompt = ["a pair of shoes."]

goal_action = grounded_graspnet.step(obs, prompt)



