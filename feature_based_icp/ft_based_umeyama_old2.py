import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import argparse
import os
import glob


def estimate_scale_from_depths(depth_rel: np.ndarray, depth_real: np.ndarray,
                               foreground_mask: np.ndarray, method='l2')->float:
    """
    obtain scale between relative depth and real depth

    Params:
    depth_rel: np.ndarray, relative depth
    depth_real: np.ndarray, real depth

    Returns:
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
        raise ValueError(f"Too few effective pixels! Valid: {len(rel_valid)}")

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


def Kabsch(P, Q, with_scale=True):
    """
    Kabsch similarity alignment: Q ≈ s * R @ P + t
    Args:
        P: (N,3) source points
        Q: (N,3) target points
    Returns:
        s: scalar
        R: (3,3) rotation
        t: (3,) translation
        err: mean point error ||Q - (sRP + t)||
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

    if with_scale:
        varP = (X**2).sum() / N
        s = D.sum() / varP
    else:
        s = 1.0

    t = muQ - s * (R @ muP)

    Q_hat = (s * (R @ P.T)).T + t
    err = np.mean(np.linalg.norm(Q - Q_hat, axis=1))
    print(f"scale: {s}, rotation:\n{R}, translation: {t}, mean error: {err}")
    return s, R, t, err

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
    s_final, R_final, t_final, err = Kabsch(pts1, pts2, with_scale=False)
    # Q = s * R @ P + t -> Q / s = R @ P + t / s
    t_final = t_final/s_final
    s_final = 1/s_final
    print(f"Kabsch error: {err/np.abs(pts2).mean()}")

    return s_final, R_final, t_final, err

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str, default='./pcd_data') # path to the source image
    parser.add_argument('--real-sim', type=str, default='sim') # path to the source image
    
    args = parser.parse_args()
    
    if args.real_sim == 'sim':
        intrinsics = np.array([[703.3542416,    0.       ,  256.       ],
        [   0.       , 703.3542416,  256.       ],
        [   0.       ,    0.       ,    1.       ]])
    else:
        intrinsics = np.array([[1164.5043680682502,    0.       , 923.0195937861359],
        [   0.       , 1164.5043680682502,  533.7860687031805],
        [   0.       ,    0.       ,    1.       ]])

    # img1_path = 'apple_data/1.png' # path to the source image
    # img1 = Image.open(img1_path).convert('RGB')
    # img_size = img1.size
    img_size = (200, 200) # target image size

    # If use 16-bit raw diparity
    # dpt_ini = 'apple_data/1_disp.png'
    # dpt_tgt =  'apple_data/2_disp.png'
    # dpt_ini = raw16_disp_to_depth(dpt_ini, fx=intrinsics[0, 0], baseline=0.12, img_size=img_size)
    # dpt_tgt = raw16_disp_to_depth(dpt_tgt, fx=intrinsics[0, 0], baseline=0.12, img_size=img_size)
    
    # If use Depth-Anything generated metric depth
    # dpt_ini = os.path.join(args.img_path, '1_dpt.png') # path to the source image
    # dpt_tgt = os.path.join(args.img_path, '2_dpt.png') # path to the target image
    # dpt_ini = os.path.join(args.img_path, 'original_depth.png') # path to the source image
    # dpt_tgt = os.path.join(args.img_path, 'edited_depth.png') # path to the target image
    # dpt_ini = np.array(Image.open(dpt_ini).convert('L').resize(img_size))
    # dpt_tgt = np.array(Image.open(dpt_tgt).convert('L').resize(img_size))
    dpt_ini = os.path.join(args.img_path, 'dpt_init.npy')
    dpt_tgt = os.path.join(args.img_path, 'dpt_goal.npy')
    # dpt_ini = np.load(dpt_ini)
    # dpt_tgt = np.load(dpt_tgt)
    # plt.imshow(dpt_ini)
    # plt.colorbar()
    # plt.savefig(os.path.join(args.img_path, 'depth_original_vis.png'), bbox_inches='tight', pad_inches=0)
    # plt.close()
    # plt.imshow(dpt_tgt)
    # plt.colorbar()
    # plt.savefig(os.path.join(args.img_path, 'depth_edited_vis.png'), bbox_inches='tight', pad_inches=0)
    # plt.close()
    dpt_ini = np.array(Image.fromarray(np.load(dpt_ini)).resize(img_size))
    dpt_tgt = np.array(Image.fromarray(np.load(dpt_tgt)).resize(img_size))
    plt.imshow(dpt_ini)
    plt.colorbar()
    plt.savefig(os.path.join(args.img_path, 'dpt_init.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.imshow(dpt_tgt)
    plt.colorbar()
    plt.savefig(os.path.join(args.img_path, 'dpt_goal.png'), bbox_inches='tight', pad_inches=0)
    plt.close()

    seg_ini = np.array(Image.open(os.path.join(args.img_path, 'mask_init.png')).convert('1').resize(img_size))
    seg_tgt = np.array(Image.open(os.path.join(args.img_path, 'mask_goal.png')).convert('1').resize(img_size))

    pattern = os.path.join(args.img_path, 'depth_*.npy')

    # 3. Use glob.glob() to find all matching files, which returns a list
    depth_files = glob.glob(pattern)

    # 4. Check if the list is empty, then load the first matched file
    if depth_files:
        depth_file_path = depth_files[0] # get the full path of the first file
        print(f"Found and loading depth file: {depth_file_path}")
        real_depth = np.load(depth_file_path)
    else:
        # If no file found, print an error and exit to avoid further issues
        print(f"Error: No file matching pattern '{pattern}' was found.")
        exit()

    # real_depth = np.load(os.path.join(args.img_path, 'depth_*.npy'))
    if real_depth.shape == (1080, 1920):
        real_depth = real_depth[:,:1440]
    real_depth = np.array(Image.fromarray(real_depth).resize(img_size))
    real_depth_mean = real_depth[seg_ini>0.5].mean()
    
    # scale = estimate_scale_from_depths(depth_rel=dpt_ini, depth_real=real_depth, 
    #                                    foreground_mask=seg_ini, method='l2')
    
    # rescale dpt_tgt to avoid extreme large or small scales (too small or too large will decrease accuracy)
    valid_mask = seg_tgt > 0.5
    if np.count_nonzero(valid_mask) > 0:
        tgt_mean = float(np.nanmean(dpt_tgt[valid_mask]))
    else:
        tgt_mean = float(np.nanmean(dpt_tgt))
    if not np.isfinite(tgt_mean) or tgt_mean <= 1e-8:
        print("[Warning] Invalid target depth mean for scaling. Skipping rescale.")
    else:
        dpt_tgt = (dpt_tgt / tgt_mean) * (real_depth_mean if np.isfinite(real_depth_mean) and real_depth_mean > 0 else 1.0)

    point_wise_matching = pickle.load(open(os.path.join(args.img_path, 'point_wise_matching.pkl'), 'rb'))

    # robust ICP for transformation
    s, R, t, err = compute_similarity_from_matchdict(depth1=real_depth, depth2=dpt_tgt, 
                                                     M=point_wise_matching, K=intrinsics,
                                                     seg1=seg_ini, seg2=seg_tgt)
    print(R, t)
    pickle.dump((R, t), open(os.path.join(args.img_path, 'transformation.pkl'), 'wb'))