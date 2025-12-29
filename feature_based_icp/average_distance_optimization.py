import torch
from PIL import Image
import numpy as np
import cv2
from lietorch import SE3, LieGroupParameter
import tqdm
import viser
import pickle
import time

def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion (w, x, y, z) to a rotation matrix (3x3).
    """
    w, x, y, z = q
    R = torch.tensor([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                      [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                      [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    return R

def normalize_quaternion(q):
    """
    Normalize the quaternion to ensure it has unit length (||q|| = 1).
    """
    norm = torch.norm(q)
    return q / norm  # Ensure q is a unit quaternion

def apply_transform(scale, rotation, translation, points):
    """
    Apply the SE(3) transformation to a set of points.
    """
    R = quaternion_to_rotation_matrix(rotation)
    return scale * torch.matmul(R, points.T).T + translation

def compute_average_distance(points1: torch.Tensor, points2: torch.Tensor)->torch.Tensor:
    """
    Compute average distance between 2 point clouds.
    Args:
        points1: torch.Tensor, shape (N, 3)
        points2: torch.Tensor, shape (M, 3)
    Returns:
        average_dist: average distance between points1 and points2
    """
    dist_mat = torch.cdist(points1, points2, p=2)  # shape (N, M)
    
    return dist_mat.mean()

def optimize_poses(points1: torch.Tensor, points2: torch.Tensor, num_iters: int = 50000):
    """
    Optimize the SE3 transformation between 2 point clouds
    Args:
        points1 (torch.Tensor): [N, 3], point cloud of initial scene
        points2 (torch.Tensor): [M, 3], point cloud of target scene
    Returns:
        s: float, scale factor
        R: (3,3), rotation matrix
        t: (3,), translation vector
    """
    
    pose_init = torch.as_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cuda().requires_grad_(True)
    s = torch.tensor(1.0, requires_grad=True)  # scale
    P = LieGroupParameter(SE3(pose_init[None]))
    l = [{'params': [P], 'lr': 1e-3, "name": "R"},
         {'params': [s], 'lr': 1e-5, "name": "s"}]
    optimizer = torch.optim.Adam(l, lr=1e-4)
    pred_points2 = P.retr().act(points1)
    pbar = tqdm.tqdm(total=num_iters, desc="Optimizing SE3 transformation")
    for iter in range(num_iters):
        optimizer.zero_grad()
        pred_points2 = P.retr().act(points1)
        avg_dist = compute_average_distance(pred_points2, points2)
        loss = avg_dist # + 1/(s+1e-4)
        loss.backward()
        optimizer.step()
        if iter % 100 == 0:
            pbar.update(100)
            pbar.set_postfix({"loss": loss.item(), "avg_dist": avg_dist.item(), "scale": s.item()})
    # Extract the optimized parameters
    s = s.item()
    Rt = P.retr().matrix().squeeze().detach()
    pickle.dump((s, Rt[:3, :3], Rt[:3, 3]), open('optimized_transformation.pkl', 'wb'))
    
    return s, Rt[:3, :3], Rt[:3, 3]
    
        
    
if __name__ == "__main__":
    
    f = 50
    
    # Extract points from the image
    img1_path = '4o_data/table_11_Color.png' # path to the source image
    img1 = Image.open(img1_path).convert('RGB')
    img2_path = '4o_data/table_12_Color.png' # path to the target image
    img2 = Image.open(img2_path).convert('RGB')
    
    seg1_path = '4o_data/table_11_mask.png' # path to the source image segmentation
    seg1 = np.array(Image.open(seg1_path).convert('1')) # [H, W]
    seg2_path = '4o_data/table_12_mask.png' # path to the target image segmentation
    seg2 = np.array(Image.open(seg2_path).convert('1'))

    dpt1_path = '4o_data/table_11_dpt.png'
    dpt2_path = '4o_data/table_12_dpt.png'
    dpt1 = cv2.imread(dpt1_path, cv2.IMREAD_GRAYSCALE)
    dpt2 = cv2.imread(dpt2_path, cv2.IMREAD_GRAYSCALE)  # [H, W]
    
    height, width = dpt1.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    dpt1 = dpt1.astype(float) / 255.0
    points1 = np.zeros((height * width, 3))
    points1[:, 0] = (x.flatten() - width/2) / f * dpt1.flatten()
    points1[:, 1] = (y.flatten() - height/2) / f * dpt1.flatten()
    points1[:, 2] = dpt1.flatten()
    points1 = torch.from_numpy(points1[seg1.flatten()>0.5]).cuda().float()  # [N, 3]
    colors1 = np.array(img1).reshape((-1, 3))[seg1.flatten()>0.5]
    
    dpt2 = dpt2.astype(float) / 255.0
    points2 = np.zeros((height * width, 3))
    points2[:, 0] = (x.flatten() - width/2) / f * dpt2.flatten()
    points2[:, 1] = (y.flatten() - height/2) / f * dpt2.flatten()
    points2[:, 2] = dpt2.flatten()
    points2 = torch.from_numpy(points2[seg2.flatten()>0.5]).cuda().float()
    colors2 = np.array(img2).reshape((-1, 3))[seg2.flatten()>0.5]
    
    print(f"points1 shape: {points1.shape}, points2 shape: {points2.shape}")
    training = True

    # Optimize the SE3 transformation
    if training:
        s, R, t = optimize_poses(points1, points2)
    else:
        s, R, t = pickle.load(open('optimized_transformation.pkl', 'rb'))
    display1 = (R@points1[...,None]+t[...,None])[...,0].cpu().numpy()
    display2 = points2.cpu().numpy()
    
    server = viser.ViserServer()
    print("启动可视化服务器，请在浏览器中访问: http://localhost:8080")
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # 创建点云可视化
        with client.atomic():
            # 设置场景的上方向为+Z
            client.scene.set_up_direction("+z")
            
            # 添加环境光
            client.scene.add_light_ambient(
                name="ambient_light",
                intensity=0.7,
                color=(255, 255, 255)
            )
            
            # 添加定向光
            client.scene.add_light_directional(
                name="directional_light",
                intensity=0.5,
                color=(255, 255, 255),
                position=(5.0, 5.0, 5.0)  # 调整光源位置
            )
            
            # 添加点云
            client.scene.add_point_cloud(
                name="pred_point_cloud",
                points=display1,
                colors=colors1,
                point_size=0.02,  # 减小点的大小
                point_shape='circle',
            )
            
            client.scene.add_point_cloud(
                name="target_point_cloud",
                points=display2,
                colors=colors2,
                point_size=0.02,  # 减小点的大小
                point_shape='circle',
            )
            
            # 添加网格 - 调整大小以匹配点云
            client.scene.add_grid(
                name="grid",
                width=4.0,
                height=4.0,
                width_segments=10,
                height_segments=10,
            )
            
            # 设置相机视角 - 调整为更近的视角
            client.camera.position = (3.0, 3.0, 3.0)
            client.camera.look_at = (0.0, 0.0, 0.0)
            client.camera.up = (0.0, 0.0, 1.0)
    
    # 保持服务器运行
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n关闭服务器...")
        server.close()