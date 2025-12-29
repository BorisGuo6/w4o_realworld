import numpy as np
import cv2
import viser
import viser.transforms as tf
import os
import time
import pickle
from PIL import Image
import plotly.graph_objects as go


def raw16_disp_to_depth(raw_disp: str, baseline: float, img_size, fx: float=703.3542416, scale: float=256.0, invalid_raw=0)->np.ndarray:
    """
    Transform 16-bit raw disparity to depth map.
      raw_disp = disparity_pix * scale
      depth = fx * baseline / disparity_pix

    Parameters:
    raw_disp: str, path to the raw disparity image
    fx: float, focal length in pixels
    baseline: float, baseline in meters
    scale: float, scale factor for disparity
    invalid_raw: int, value for invalid disparity

    Returns:
    depth_map: np.ndarray, standardized depth map
    """
    img = Image.open(raw_disp).resize(img_size)
    # make sure is 16-bit raw image
    if img.mode != 'I;16' and img.mode != 'I':
        raise ValueError(f"Unexpected image mode: {img.mode}. Expected 16-bit grayscale.")
    raw_disp = np.array(img, dtype=np.uint16)
    
    raw = raw_disp.astype(np.float32)
    mask = raw > invalid_raw

    # Transforma the disparity to standardized disparity
    disp_pix = np.zeros_like(raw, dtype=np.float32)
    disp_pix[mask] = raw[mask] / scale

    # disparity to depth
    depth = np.zeros_like(raw, dtype=np.float32)
    depth[mask] = fx * baseline / disp_pix[mask]
    depth[~mask] = np.nan
    return depth

def create_depth_point_cloud(depth_path: str, tgt_depth_path: str, img_size: tuple[int, int]=(960, 960), f: float=703.3542416):
    M = pickle.load(open('point_wise_matching.pkl', 'rb'))
    num_flow = 50
    indices = np.random.choice(len(M), num_flow, replace=False)
    src = []
    tgt = []
    for i in range(num_flow):
        src.append(list(M.keys())[indices[i]]) #(x, y)
        tgt.append(list(M.values())[indices[i]]) #(y, x)
    # 检查文件是否存在
    if not os.path.exists(depth_path):
        print(f"错误：深度图文件不存在: {depth_path}")
        return
    
    # 读取深度图
    depth = raw16_disp_to_depth(depth_path, 0.1, img_size)
    depth = depth/depth.max()
        
    print(f"深度图尺寸: {depth.shape}")
    
    # 获取图像尺寸
    height, width = depth.shape
    
    # 创建坐标网格
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 创建点云 - 调整缩放使点云更紧凑
    points = np.zeros((height * width, 3))
    points[:, 0] = (x.flatten() - width/2) / width * 2
    points[:, 1] = (y.flatten() - height/2) / height *2
    points[:, 2] = depth.flatten() * 2
    
    seg_1 = seg1.flatten() #[N]
    seg_2 = seg2.flatten()
    # points = points[seg_1 > 0.5]
    
    # 使用原色
    colors = img1.reshape(-1, 3)# [seg_1 > 0.5]  # [N, 3]
    colors2 = img2.reshape(-1, 3)[seg_2 > 0.5]  # [N, 3]
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # 读取深度图
    tgt_depth = raw16_disp_to_depth(tgt_depth_path, 0.1, img_size)
    tgt_depth = tgt_depth/tgt_depth.max()
    
    # target point cloud
    points2 = np.zeros((height * width, 3))
    points2[:, 0] = (x.flatten() - width/2) / width * 2
    points2[:, 1] = (y.flatten() - height/2) / height * 2
    points2[:, 2] = tgt_depth.flatten() * 2
    points2 = points2[seg_2 > 0.5]
    
    points = np.concatenate([points, points2], axis=0)  # [2N, 3]
    colors = np.concatenate([colors, colors2], axis=0)/255.0  # [2N, 3]
    print(colors.shape)
    print(points.shape)

    # 创建flow
    src_points = np.zeros((len(src), 3))
    src_colors = np.zeros((len(src), 3))
    for i in range(len(src)):
        src_points[i,0] = (src[i][1] - width/2) / width * 2
        src_points[i,1] = (src[i][0] - height/2) / height * 2
        src_points[i,2] = depth[src[i][0], src[i][1]] * 2
        src_colors[i] = img1[src[i][0], src[i][1], :]/255.0  # [N, 3]

    tgt_points = np.zeros((len(tgt), 3))
    tgt_colors = np.zeros((len(tgt), 3))
    for i in range(len(tgt)):
        tgt_points[i,0] = (tgt[i][1] - width/2) / width * 2
        tgt_points[i,1] = (tgt[i][0] - height/2) / height * 2
        tgt_points[i,2] = tgt_depth[tgt[i][0], tgt[i][1]] * 2
        tgt_colors[i] = img2[tgt[i][0], tgt[i][1], :]/255.0  # [N, 3]
    assert tgt_points.shape[0] == src_points.shape[0], "点云数量不匹配"
    print(f"点云数量: {tgt_points.shape[0]}")
    
    flow = np.stack([src_points, tgt_points], axis=1)  # [N, 2, 3]
    flow_colors = np.stack([src_colors, tgt_colors], axis=1)  # [N, 2, 3]
    
    # 创建viser服务器
    server = viser.ViserServer()
    print("启动可视化服务器，请在浏览器中访问: http://localhost:8080")
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # 创建点云可视化
        with client.atomic():
            # 设置场景的上方向为+Z
            client.scene.set_up_direction("+z")
            # 添加点云
            client.scene.add_point_cloud(
                name="point_cloud",
                points=points,
                colors=colors,
                point_size=0.01,  # 减小点的大小
                point_shape='circle',
            )
            
            # 添加flow
            client.scene.add_line_segments(
                name="flow",
                points=flow,
                colors=flow_colors,
                line_width=0.5,  # 调整线宽
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

# 使用示例
depth_path = "apple_data/1_disp.png"
tgt_depth_path = "apple_data/2_disp.png"
img_size = (960, 960) # 目标图像大小

seg1_path = 'apple_data/1_mask.png' # path to the source image segmentation
seg1 = np.array(Image.open(seg1_path).convert('1').resize(img_size)) # [H, W]
seg2_path = 'apple_data/2_mask.png' # path to the target image segmentation
seg2 = np.array(Image.open(seg2_path).convert('1').resize(img_size))

img1_path = 'apple_data/1.png' # path to the source image
img1 = np.array(Image.open(img1_path).convert('RGB').resize(img_size))
img2_path = 'apple_data/2.png' # path to the target image
img2 = np.array(Image.open(img2_path).convert('RGB').resize(img_size))

# 运行交互式可视化
create_depth_point_cloud(depth_path, tgt_depth_path, img_size=img_size)
