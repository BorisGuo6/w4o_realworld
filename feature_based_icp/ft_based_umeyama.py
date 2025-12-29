import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import glob
import viser
import time
from sklearn.cluster import KMeans

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
    mask = (rel > 0.5) & (real > 0.5) #& (foreground > 0.5)
    rel_valid = rel[mask]
    real_valid = real[mask]
    
    if len(rel_valid) < 10:
        raise ValueError(f"To few effective pixels! Valid: {len(rel_valid)}")

    # if method == 'l2':
    #     if rel_valid.mean() <= real_valid.mean():
    #         s = np.sum(real_valid * rel_valid) / np.sum(rel_valid**2)
    #     else:
    #         s = np.sum(real_valid**2) /np.sum(real_valid * rel_valid)
    # elif method == 'median':
    #     s = np.median(real_valid / rel_valid)
    # else:
    #     raise ValueError("l2 or median must be chosen!")
    s = np.sum((real_valid - real_valid.mean()) * (rel_valid - rel_valid.mean())) / np.sum((rel_valid - rel_valid.mean())**2)
    b = real_valid.mean() - s * rel_valid.mean()
    print(f"Estimated scale: {s}, bias: {b}, num of valid pixels: {len(rel_valid)}")
    return s, b

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

def Kabsch(P, Q, with_scale=True):
    """
    Kabsch similarity alignment: s * Q ≈ R @ P + t
    Args:
        P: (N,3) source points
        Q: (N,3) target points
    Returns:
        s: scalar
        R: (3,3) rotation
        t: (3,) translation
        err: mean point error ||sQ - (RP + t)||
    """
    assert P.shape == Q.shape
    N = P.shape[0]

    muP = P.mean(axis=0)
    muQ = Q.mean(axis=0)
    X = P - muP
    Y = Q - muQ
    Sigma = (X.T @ Y) / N

    U, D, Vt = np.linalg.svd(Sigma)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        D[-1] *= -1
    # R = np.eye(3)

    if with_scale:
        varP = (X**2).mean()
        varQ = (Y**2).mean()
        s = np.sqrt(varP / varQ)
    else:
        s = 1.0

    t = s*muQ - (R @ muP)

    Q_hat = (R @ P.T).T + t
    rel_err = np.linalg.norm(s*Q - Q_hat, axis=1).mean()
    print(f"scale: {s}, rotation:\n{R}, translation: {t}, mean error: {rel_err}")
    print(muP-muQ)
    return s, R, t, rel_err, P, Q, Q_hat

def compute_similarity_from_matchdict(M, depth1: np.ndarray, depth2: np.ndarray, K: np.ndarray,
                                      seg1: np.ndarray, seg2: np.ndarray):
    """
    Use Kabsch algorithm to estimate Sim(3) transformation between two point clouds based on the point-wise matching dictionary M.  
    Params:
        M: dict, key: (x1, y1), value: (x2, y2)
        depth1: np.ndarray, depth map of the first image, real
        depth2: np.ndarray, depth map of the second image, subgoal
        K: np.ndarray, camera intrinsic matrix
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
        if np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= img_size[0]*0.05:
            continue
        d1 = depth1[x1, y1]
        d2 = depth2[x2, y2]
        if seg1[x1, y1] < 0.5 or seg2[x2, y2] < 0.5:
            continue
        # u1 = np.array([(x1 + cx) / fx, (y1 + cy) / fy, 1.0])
        # u2 = np.array([(x2 + cx) / fx, (y2 + cy) / fy, 1.0])
        u1 = np.array([(y1 + cx) / fx, (x1 + cy) / fy, 1.0])
        u2 = np.array([(y2 + cx) / fx, (x2 + cy) / fy, 1.0])
        pts1.append(d1 * u1)
        pts2.append(d2 * u2)
    pts1 = np.stack(pts1, axis=0)
    pts2 = np.stack(pts2, axis=0)
    label1 = KMeans(n_clusters=2, random_state=0).fit(pts1)
    label2 = KMeans(n_clusters=2, random_state=0).fit(pts2)
    main_label1 = np.argmax(np.bincount(label1.labels_))
    main_label2 = np.argmax(np.bincount(label2.labels_))
    filter1 = label1.labels_ == main_label1
    filter2 = label2.labels_ == main_label2
    filter = filter1 # & filter2
    pts1 = pts1[filter]
    pts2 = pts2[filter]
    print(np.linalg.norm(pts1-pts2, axis=1))
    N = pts1.shape[0]
    if N < 3:
        raise ValueError("At least 3 points are required for RANSAC.")
    s_final, R_final, t_final, err, P, Q, Q_hat = Kabsch(pts1, pts2, with_scale=False)

    return s_final, R_final, t_final, err, P, Q, Q_hat

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str, default='./pcd_data') # path to the source image
    parser.add_argument('--real-sim', type=str, default='sim') # path to the source image
    
    args = parser.parse_args()
    
    # if args.real_sim == 'sim':
    #     intrinsics = -np.array([[-703.3542416,    0.       ,  256.       ],
    #     [   0.       , -703.3542416,  256.       ],
    #     [   0.       ,    0.       ,    1.       ]])
    # else:
    #     intrinsics = -np.array([[-1164.5043680682502,    0.       , 923.019597861359],
    #     [   0.       , -1164.5043680682502,  533.7860687031805],
    #     [   0.       ,    0.       ,    1.       ]])

    # img1_path = 'apple_data/1.png' # path to the source image
    # img1 = Image.open(img1_path).convert('RGB')
    # img_size = img1.size
    img_size = (200, 200) # 目标图像大小
    if args.real_sim == 'sim':
        intrinsics = -np.array([[-703.3542416,    0.       ,  256.       ],
        [   0.       , -703.3542416,  256.       ],
        [   0.       ,    0.       ,    1.       ]])
    else:
        intrinsics = -np.array([[-1164.5043680682502/923.019597861359*img_size[0]/2,    0.       , img_size[0]/2],
        [   0.       , -1164.5043680682502/533.7860687031805*img_size[1]/2,  img_size[1]/2],
        [   0.       ,    0.       ,    1.       ]])

    dpt_ini = os.path.join(args.img_path, 'dpt_init.png') # path to the source image
    dpt_ini = np.array(Image.open(dpt_ini).convert('L').resize(img_size))
    dpt_tgt = os.path.join(args.img_path, 'dpt_goal.png') # path to the target image
    dpt_tgt = np.array(Image.open(dpt_tgt).convert('L').resize(img_size))

    plt.imshow(dpt_ini)
    plt.colorbar()
    plt.savefig("dpt_ini.png")
    plt.close()

    plt.imshow(dpt_tgt)
    plt.colorbar()
    plt.savefig("dpt_tgt.png")
    plt.close()

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
    plt.imshow(real_depth)
    plt.colorbar()
    plt.savefig("real_depth.png")
    plt.close()
    
    dpt_scale, dpt_bias = estimate_scale_from_depths(depth_rel=dpt_ini, depth_real=real_depth, 
                                       foreground_mask=seg_ini, method='l2')
    
    # rescale dpt_tgt to avoid extreme large or small scales (too small or too large will decrease accuracy)
    # dpt_tgt = (dpt_tgt/dpt_tgt[seg_tgt<=0.5].mean())*real_depth_mean
    dpt_tgt_resized = dpt_tgt * dpt_scale + dpt_bias
    dpt_ini_resized = dpt_ini * dpt_scale + dpt_bias

    point_wise_matching = pickle.load(open(os.path.join(args.img_path, 'point_wise_matching.pkl'), 'rb'))

    # robust ICP for transformation
    s, R, t, err, P, Q, Q_hat = compute_similarity_from_matchdict(depth1=dpt_ini_resized, depth2=dpt_tgt_resized, 
                                                     M=point_wise_matching, K=intrinsics,
                                                     seg1=seg_ini, seg2=seg_tgt)
    print(R, t)
    Q_hat = R @ P.mean(axis=0) + t
    print(Q_hat-P.mean(axis=0))
    print(P.mean(axis=0))
    t = Q.mean(axis=0) - P.mean(axis=0)
    pickle.dump((R, t), open(os.path.join(args.img_path, 'transformation.pkl'), 'wb'))