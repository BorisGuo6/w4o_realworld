import numpy as np
import fpsample
from termcolor import cprint

def fpsample_pc(scene_pc, scene_colors=None, n_save_points=512, verbose=False):
    all_sampled_points = np.zeros((n_save_points, 3), dtype=np.float32)
    if scene_colors is not None:
        all_sampled_colors = np.zeros((n_save_points, 3), dtype=np.uint8)
    if len(scene_pc) > n_save_points:
        # Use FPS to downsample points
        point_idx = fpsample.bucket_fps_kdline_sampling(
            scene_pc, n_save_points, h=3
        )
        all_sampled_points[:] = scene_pc[point_idx]
        if scene_colors is not None:
            all_sampled_colors[:] = scene_colors[point_idx]
    else:
        if verbose:
            cprint(f"Fewer points than requested: {len(scene_pc)} < {n_save_points}", "yellow")
        # If we have fewer points than requested, pad with randomly sampled points
        n_missing = n_save_points - len(scene_pc)
        random_indices = np.random.choice(len(scene_pc), size=n_missing, replace=True)
        
        # First assign all existing points
        all_sampled_points[:len(scene_pc)] = scene_pc
        if scene_colors is not None:
            all_sampled_colors[:len(scene_pc)] = scene_colors
        
        # Then pad with randomly sampled points
        all_sampled_points[len(scene_pc):] = scene_pc[random_indices]
        if scene_colors is not None:
            all_sampled_colors[len(scene_pc):] = scene_colors[random_indices]
    if scene_colors is not None:
        return all_sampled_points, all_sampled_colors
    else:
        return all_sampled_points