import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import glob
import time, viser

def estimate_scale_from_depths(depth_rel: np.ndarray, depth_real: np.ndarray,
                               foreground_mask: np.ndarray, method='l2')->float:
    """
    obtain scale between relative depth and real depth

    Params:
    depth_rel: np.ndarray, relative depth
    depth_real: np.ndarray, real depth

    返回：
    - s: float, scale factor, depth_real = s * depth_rel
    - num: int, number of valid pixel pairs
    """
    rel = depth_rel.flatten()
    real = depth_real.flatten()
    foreground = foreground_mask.flatten()
    mask = (rel > 0) & (real > 0) & (foreground > 0.5)
    rel_valid = rel[mask]
    real_valid = real[mask]
    
    if len(rel_valid) < 10:
        raise ValueError(f"To few effective pixels! Valid: {len(rel_valid)}")

    if method == 'l2':
        if rel_valid.mean() <= real_valid.mean():
            s = np.sum(real_valid * rel_valid) / np.sum(rel_valid**2)
        else:
            s = np.sum(real_valid**2) /np.sum(real_valid * rel_valid)
    elif method == 'median':
        s = np.median(real_valid / rel_valid)
    else:
        raise ValueError("l2 or median must be chosen!")
    print(f"scale error:  {np.abs(real_valid - s * rel_valid).mean() / np.abs(real_valid).mean()}")
    return s

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

def compute_similarity_from_matchdict(M, depth1: np.ndarray, depth2: np.ndarray, K: np.ndarray,
                                      seg1: np.ndarray, seg2: np.ndarray):
    """
    Use Umeyaama algorithm to estimate the similarity transform (s, R, t) between two point clouds based on the point-wise matching dictionary M.  
    Params:
        M: dict, key: (x1, y1), value: (x2, y2)
        depth1: np.ndarray, depth map of the first image, real
        depth2: np.ndarray, depth map of the second image, subgoal
        K: np.ndarray, camera intrinsic matrix
        ransac_iters: int, number of RANSAC iterations
        inlier_thr: float, inlier threshold for RANSAC
    Returns:
        s, R, t: scale, rotation and transitions
    """
    H, W = depth1.shape
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    pts1, pts2 = [], []
    for (x1, y1), (x2, y2) in M.items():
        d1 = depth1[x1, y1]
        d2 = depth2[x2, y2]
        if seg1[x1, y1] < 0.5 or seg2[x2, y2] < 0.5:
            continue
        u1 = np.array([(x1 + cx) / fx, (y1 + cy) / fy, 1.0])
        u2 = np.array([(x2 + cx) / fx, (y2 + cy) / fy, 1.0])
        pts1.append(d1 * u1)
        pts2.append(d2 * u2)

    pts1 = np.stack(pts1, axis=0)
    pts2 = np.stack(pts2, axis=0)
    N = pts1.shape[0]
    if N < 3:
        raise ValueError("At least 3 points are required for RANSAC.")

    # Umeyama estimation, note that since P is lifted from real depth, so the formula changes to s*Q=R@P+t
    def umeyama(P, Q, with_scale=True):
        N = P.shape[0]
        muP = P.mean(axis=0)
        muQ = Q.mean(axis=0)
        P0 = P - muP
        Q0 = Q - muQ
        
        H = P0.T @ Q0
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = (R @ muP) - muQ
        if with_scale:
            num = np.sum((R @ P.T + t[:, None])**2)
            denom = np.sum(Q.T**2)
            s = np.sqrt(num / denom)
        else:
            s = 1.0
        print(Q.shape, R.shape, P.T.shape, t[:, None].shape)
        err = np.sqrt(np.sum((s * Q - (R @ P.T + t[...,None]).T)**2) / N)
        
        return s, R, t, err

    # Umeyama estimation on 2 point clouds
    s_final, R_final, t_final, err = umeyama(pts1, pts2, with_scale=True)
    print(f"Umeyama error: {err/np.abs(pts1).mean()}")

    return s_final, R_final, t_final, err

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str, default='./pcd_data') # path to the source image
    parser.add_argument('--real-sim', type=str, default='sim') # path to the source image
    
    args = parser.parse_args()
    
    if args.real_sim == 'sim':
        intrinsics = -np.array([[-703.3542416,    0.       ,  256.       ],
        [   0.       , -703.3542416,  256.       ],
        [   0.       ,    0.       ,    1.       ]])
    else:
        intrinsics = -np.array([[-1164.5043680682502,    0.       , 923.0195937861359],
        [   0.       , -1164.5043680682502,  533.7860687031805],
        [   0.       ,    0.       ,    1.       ]])

    # img1_path = 'apple_data/1.png' # path to the source image
    # img1 = Image.open(img1_path).convert('RGB')
    # img_size = img1.size
    img_size = (200, 200) # 目标图像大小

    # If use 16-bit raw diparity
    # dpt_ini = 'apple_data/1_disp.png'
    # dpt_tgt =  'apple_data/2_disp.png'
    # dpt_ini = raw16_disp_to_depth(dpt_ini, fx=intrinsics[0, 0], baseline=0.12, img_size=img_size)
    # dpt_tgt = raw16_disp_to_depth(dpt_tgt, fx=intrinsics[0, 0], baseline=0.12, img_size=img_size)
    
    # If use Depth-Anything generated metric depth
    # dpt_ini = os.path.join(args.img_path, '1_dpt.png') # path to the source image
    # dpt_tgt = os.path.join(args.img_path, '2_dpt.png') # path to the target image
    dpt_ini = os.path.join(args.img_path, 'dpt_init.png') # path to the source image
    dpt_tgt = os.path.join(args.img_path, 'dpt_goal.png') # path to the target image
    dpt_ini = np.array(Image.open(dpt_ini).convert('L').resize(img_size))
    dpt_tgt = np.array(Image.open(dpt_tgt).convert('L').resize(img_size))

    seg_ini = np.array(Image.open(os.path.join(args.img_path, 'mask_init.png')).convert('1').resize(img_size))
    seg_tgt = np.array(Image.open(os.path.join(args.img_path, 'mask_goal.png')).convert('1').resize(img_size))

    pattern = os.path.join(args.img_path, 'depth_*.npy')

    # 3. 使用 glob.glob() 查找所有匹配的文件，它会返回一个列表
    depth_files = glob.glob(pattern)

    # 4. 检查列表是否为空，然后加载第一个找到的文件
    if depth_files:
        depth_file_path = depth_files[0] # 获取找到的第一个文件的完整路径
        print(f"Found and loading depth file: {depth_file_path}")
        real_depth = np.load(depth_file_path)
    else:
        # 如果没有找到文件，打印错误信息并退出，避免后续代码出错
        print(f"Error: No file matching pattern '{pattern}' was found.")
        exit() # 或者 raise FileNotFoundError("没有找到深度文件")

    # real_depth = np.load(os.path.join(args.img_path, 'depth_*.npy'))
    if real_depth.shape == (1080, 1920):
        real_depth = real_depth[:,:1440]
    real_depth = np.array(Image.fromarray(real_depth).resize(img_size))
    real_depth_mean = real_depth[seg_ini>0.5].mean()
    
    # scale = estimate_scale_from_depths(depth_rel=dpt_ini, depth_real=real_depth, 
    #                                    foreground_mask=seg_ini, method='l2')
    
    # rescale dpt_tgt to avoid extreme large or small scales (too small or too large will decrease accuracy)
    dpt_tgt = (dpt_tgt/dpt_tgt[seg_tgt>0.5].mean())*real_depth_mean

    point_wise_matching = pickle.load(open(os.path.join(args.img_path, 'point_wise_matching.pkl'), 'rb'))

    # robust ICP for transformation
    s, R, t, err = compute_similarity_from_matchdict(depth1=real_depth, depth2=dpt_tgt, 
                                                     M=point_wise_matching, K=intrinsics,
                                                     seg1=seg_ini, seg2=seg_tgt)
    R = np.eye(3)
    img_ini = np.array(Image.open(os.path.join(args.img_path, 'rgb_init.png')).convert('RGB').resize(img_size))
    img_goal = np.array(Image.open(os.path.join(args.img_path, 'rgb_goal.png')).convert('RGB').resize(img_size))
    H, W = real_depth.shape
    grids = np.meshgrid(np.arange(W), np.arange(H)) # [H, W]
    us = grids[0].flatten()
    vs = grids[1].flatten()
    ds = real_depth.flatten()
    filter = (ds>=0.75)

    us = us[filter]
    vs = vs[filter]
    ds = ds[filter]
    xs = (us + intrinsics[0,2]) * ds / intrinsics[0,0]
    ys = (vs + intrinsics[1,2]) * ds / intrinsics[1,1]
    zs = ds
    pts = np.stack([xs, ys, zs], axis=1) # [H*W, 3]
    colors = img_ini.reshape(-1, 3)/255.0 # [H*W, 3]
    colors = colors[filter]

    seg_ini = seg_ini.flatten()[filter]
    obj_pts = pts[seg_ini>0.5]
    goal_pts = (R@obj_pts.T * s + t[:, None]).T
    goal_colors = colors[seg_ini>0.5]

    grids = np.meshgrid(np.arange(W), np.arange(H)) # [H, W]
    us = grids[0].flatten()
    vs = grids[1].flatten()
    goal_ds = dpt_tgt.flatten()
    goal_xs = (us + intrinsics[0,2]) * goal_ds / intrinsics[0,0]
    goal_ys = (vs + intrinsics[1,2]) * goal_ds / intrinsics[1,1]
    goal_zs = goal_ds
    goal_gt_pts = s*np.stack([goal_xs, goal_ys, goal_zs], axis=1) # [H*W, 3]\\

    seg_tgt = seg_tgt.flatten()
    goal_gt_pts = goal_gt_pts * s
    goal_gt_obj_pts = goal_gt_pts[seg_tgt>0.5]
    goal_gt_colors = img_goal.reshape(-1, 3)/255.0
    goal_gt_obj_colors = goal_gt_colors[seg_tgt>0.5]
    print((goal_pts-obj_pts).mean(axis=0), obj_pts.mean(axis=0))
    server = viser.ViserServer(port=8081)
    print("启动可视化服务器，请在浏览器中访问: http://localhost:8081")
    
    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        # 创建点云可视化
        with client.atomic():
            # 设置场景的上方向为+Z
            client.scene.set_up_direction("+z")
            
            # 添加点云
            client.scene.add_point_cloud(
                name="ini_scene",
                points=pts,
                colors=colors,
                point_size=0.005,  # 减小点的大小
                point_shape='circle',
            )
            client.scene.add_point_cloud(
                name="goal_obj",
                points=goal_pts,
                colors=goal_colors,
                point_size=0.005,  # 减小点的大小
                point_shape='circle',
            )
            # client.scene.add_point_cloud(
            #     name="goal_gt_obj",
            #     points=goal_gt_obj_pts,
            #     colors=goal_gt_obj_colors,
            #     point_size=0.005,  # 减小点的大小
            #     point_shape='circle',
            # )
            client.scene.add_point_cloud(
                name="goal_gt_scene",
                points=goal_gt_pts ,
                colors=goal_gt_colors,
                point_size=0.01,  # 减小点的大小
                point_shape='circle',
            )
            # 设置相机视角 - 调整为更近的视角
            client.camera.position = (0.0, 0.0, 0.0)
            client.camera.look_at = (0.0, 0.0, 1.0)
            client.camera.up = (1.0, 0.0, 0.0)

    # 保持服务器运行
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n关闭服务器...")
        server.close()