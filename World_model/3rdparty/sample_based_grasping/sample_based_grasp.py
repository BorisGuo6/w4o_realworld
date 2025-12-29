import numpy as np
from PIL import Image
import open3d as o3d
import trimesh
import yourdfpy
import sys
from pathlib import Path

PROJECT_ROOT = Path('/home/world4omni/w4o')       # should be ~/w4o
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
GRIPPER_MODEL_PATH = PROJECT_ROOT / "xarm7" / "assets" / "xarm_gripper" / "hand_open.obj"
sys.path.insert(0, str(PROJECT_ROOT))
from World4Omni_rw.tools.get_new import get_newest


def get_bbox(obj_points):
    """
    Get the oriented bounding box (OBB) of the object point cloud using Open3D.
    The bounding box is the smallest possible box containing all points and is not
    necessarily aligned with the XYZ axes.

    Args:
        obj_points (np.ndarray): Nx3 array of object points.

    Returns:
        o3d.geometry.OrientedBoundingBox: The computed oriented bounding box.
    """

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)
    
    # Compute the oriented bounding box
    obj_bbox = pcd.get_oriented_bounding_box()
    obj_bbox.color = (1, 0, 0) # Set color to red for visualization
    return obj_bbox


def filter_object_points_by_distance(initial_obj_points, min_dist=0.8, max_dist=1.8):
    """
    Filters the object point cloud by distance from the origin (camera).

    Args:
        initial_obj_points (np.ndarray): The initial object points to filter.
        min_dist (float): Minimum distance from the origin.
        max_dist (float): Maximum distance from the origin.

    Returns:
        np.ndarray: The filtered object points.
    """
    print("\n1.5. Filtering object points by distance...")
    if initial_obj_points.shape[0] > 0:
        distances = np.linalg.norm(initial_obj_points, axis=1)
        in_range_mask = (distances >= min_dist) & (distances <= max_dist)
        obj_points = initial_obj_points[in_range_mask]
        
        num_initial = len(initial_obj_points)
        num_final = len(obj_points)
        print(f"   Filtered object points from {num_initial} to {num_final} (kept points between {min_dist}-{max_dist}m).")
    else:
        obj_points = initial_obj_points # No points to filter
        print("   No object points found from mask, skipping distance filter.")
    
    return obj_points


def preprocess(scene_points, mask_image):
    """
    Preprocess the scene points and mask image to extract object points.
    It assumes the scene_points correspond to the pixels of the mask_image when flattened.

    Args:
        scene_points (np.ndarray): Mx3 array of points in the scene.
        mask_image (PIL.Image): A PIL Image object for the mask.

    Returns:
        np.ndarray: Nx3 array of points corresponding to the object.
    """
    # Convert PIL image to a numpy array
    mask_array = np.array(mask_image)
    
    # Ensure mask is 2D (in case it's a color image)
    if mask_array.ndim == 3:
        # Convert to grayscale by taking the first channel, assumes non-zero pixels are the mask
        mask_array = mask_array[:, :, 0]

    # Flatten the mask to a 1D boolean array
    # True where pixel value is not zero
    flat_mask = mask_array.flatten() > 0

    # Ensure the mask length matches the number of points
    if len(flat_mask) != len(scene_points):
        print(f"Warning: Mask size ({len(flat_mask)}) does not match point cloud size ({len(scene_points)}).")
        # Attempt to resize mask, though this might not be correct if aspect ratios differ.
        # A better solution would be to ensure data recording is consistent.
        min_len = min(len(flat_mask), len(scene_points))
        flat_mask = flat_mask[:min_len]
        scene_points = scene_points[:min_len]

    obj_points = scene_points[flat_mask]
    return obj_points


def sample_grasp(obj_points, num_grasps=200, z_offset=0.06):
    """
    Sample grasp configurations from the object point cloud.
    Grasps are sampled on the object surface, aligned with the local surface normal,
    and with a random rotation around the approach axis.

    Args:
        obj_points (np.ndarray): Nx3 array representing the object point cloud.
        num_grasps (int): The number of grasps to sample.
        z_offset (float): The distance to offset the gripper back along the approach
                          direction (z-axis) to prevent collision with the base.

    Returns:
        list[np.ndarray]: A list of 4x4 grasp pose matrices.
    """
    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(obj_points)

    # Estimate surface normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)
    
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    grasps = []
    # Randomly select points to generate grasps
    indices = np.random.choice(len(points), size=num_grasps, replace=False)

    for idx in indices:
        point = points[idx]
        normal = normals[idx]

        # 1. Define the grasp approach axis (z-axis), pointing into the surface
        z_axis = -normal
        z_axis /= np.linalg.norm(z_axis)

        # 2. Create an initial, deterministic frame perpendicular to the z-axis.
        # This gives us a starting point for our X and Y axes.
        fixed_vec = np.array([0, 1, 0])
        if np.allclose(np.abs(np.dot(z_axis, fixed_vec)), 1.0):
             fixed_vec = np.array([1, 0, 0]) # Use x-axis if z is aligned with y

        x_axis_initial = np.cross(fixed_vec, z_axis)
        x_axis_initial /= np.linalg.norm(x_axis_initial)
        
        y_axis_initial = np.cross(z_axis, x_axis_initial)
        y_axis_initial /= np.linalg.norm(y_axis_initial)

        # 3. Add random rotation around the approach axis (z-axis)
        random_angle = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(random_angle), np.sin(random_angle)
        
        # The new x and y axes are a rotation of the initial axes in their plane
        x_axis = c * x_axis_initial + s * y_axis_initial
        y_axis = -s * x_axis_initial + c * y_axis_initial

        # 4. Create the final 4x4 transformation matrix
        rotation = np.vstack([x_axis, y_axis, z_axis]).T
        transform = np.eye(4)
        transform[:3, :3] = rotation
        
        # Apply the offset: move the gripper's origin BACKWARDS from the contact point
        # opposite to the approach direction (z_axis). This ensures the fingertips
        # are at the surface, not the gripper's base.
        transform[:3, 3] = point - z_axis * z_offset
        
        grasps.append(transform)

    return grasps



def filter_grasps_by_orientation(grasps, angle_thresholds_deg):
    """
    根据抓取接近向量与世界坐标轴的角度来筛选抓取姿态。

    Args:
        grasps (list[np.ndarray]): 4x4抓取姿态矩阵的列表。
        angle_thresholds_deg (dict): 一个字典，指定每个轴的最小/最大角度（度数）。
                                     示例: {'x': (0, 90), 'z': (120, 180)}
                                     只包含你想筛选的轴。

    Returns:
        tuple: 一个元组，包含:
            - list[np.ndarray]: 有效的抓取姿态列表。
            - list[np.ndarray]: 无效的抓取姿态列表。
    """
    valid_grasps = []
    invalid_grasps = []
    
    world_axes = {
        'x': np.array([1.0, 0.0, 0.0]),
        'y': np.array([0.0, 1.0, 0.0]),
        'z': np.array([0.0, 0.0, 1.0])
    }

    for grasp_pose in grasps:
        # 抓取的接近向量是其旋转矩阵的Z轴
        approach_vector = grasp_pose[:3, 2]
        
        is_valid = True
        for axis_name, (min_angle, max_angle) in angle_thresholds_deg.items():
            if axis_name not in world_axes:
                continue
            
            world_axis = world_axes[axis_name]
            
            # 点积 = cos(theta)
            # 使用np.clip确保点积在[-1, 1]范围内，避免arccos的数学错误
            dot_product = np.clip(np.dot(approach_vector, world_axis), -1.0, 1.0)
            angle_rad = np.arccos(dot_product)
            angle_deg = np.rad2deg(angle_rad)
            
            # 检查角度是否在指定的范围内
            if not (min_angle <= angle_deg <= max_angle):
                is_valid = False
                break # 如果一个轴不满足，则无需检查其他轴
        
        if is_valid:
            valid_grasps.append(grasp_pose)
        else:
            invalid_grasps.append(grasp_pose)
            
    return valid_grasps, invalid_grasps


def check_collision(scene_points, grasps, gripper_obj_path):
    """
    Check for collisions between the gripper and the scene for each grasp.
    This version loads a single mesh file for the gripper and checks if any
    scene points are contained within the gripper mesh.

    Args:
        scene_points (np.ndarray): Mx3 array representing the entire scene point cloud.
        grasps (list[np.ndarray]): A list of 4x4 grasp pose matrices.
        gripper_obj_path (str): Path to the gripper's mesh file (e.g., .obj, .stl).

    Returns:
        list[np.ndarray]: A list of valid, collision-free 4x4 grasp pose matrices.
    """
    if not grasps:
        return []

    # Load the gripper mesh from a single file
    try:
        gripper_mesh = trimesh.load(gripper_obj_path, force='mesh')
    except Exception as e:
        print(f"Failed to load gripper mesh file: {e}")
        print(f"Please check the path: {gripper_obj_path}")
        return []

    # If the resulting mesh is empty, something went wrong with loading
    if not isinstance(gripper_mesh, trimesh.Trimesh) or gripper_mesh.is_empty:
        print(f"Warning: Loaded gripper mesh from {gripper_obj_path} is empty.")
        return []
        
    valid_grasps = []
    for grasp_pose in grasps:
        # Transform the gripper to the grasp pose
        transformed_gripper = gripper_mesh.copy()
        transformed_gripper.apply_transform(grasp_pose)

        # Check if any scene points are inside the transformed gripper mesh.
        # The .contains() method is a robust way to check for collision between a
        # watertight mesh and a point cloud. A collision occurs if any point is inside.
        if not np.any(transformed_gripper.contains(scene_points)):
            valid_grasps.append(grasp_pose)
            
    return valid_grasps


def check_grasp_quality(grasps, obj_points,  gripper_thickness=0.02, gripper_opening=0.08, gripper_depth=0.05):
    """
    Evaluates the quality of grasps based on the number of object points
    contained within the gripper's closing volume.

    Args:
        grasps (list[np.ndarray]): A list of 4x4 grasp pose matrices.
        obj_points (np.ndarray): Nx3 array of object points.
        gripper_opening (float): The width between the gripper fingers (Y-axis).
        gripper_depth (float): The depth of the gripper fingers (Z-axis).
        gripper_thickness (float): The thickness of the gripper fingers (X-axis).

    Returns:
        tuple: A tuple containing:
            - list[np.ndarray]: The list of grasps, sorted by score.
            - list[float]: A list of quality scores, sorted.
    """
    if not grasps:
        return [], []
        
    scores = []
    for grasp_pose in grasps:
        # Inverse transform to bring points into the gripper's local frame
        inv_grasp_pose = np.linalg.inv(grasp_pose)
        
        # Transform object points into gripper frame
        obj_points_homogeneous = np.hstack((obj_points, np.ones((obj_points.shape[0], 1))))
        points_in_gripper_frame = (inv_grasp_pose @ obj_points_homogeneous.T).T[:, :3]

        # Define the gripper's closing volume (a bounding box in its local frame)
        # Assuming gripper approaches along Z, closes along Y, and its thickness is along X.
        x_min, x_max = -gripper_thickness / 2, gripper_thickness / 2
        y_min, y_max = -gripper_opening / 2, gripper_opening / 2
        z_min, z_max = 0, gripper_depth # Check from gripper base to finger tips

        # Count points inside this volume
        in_x = (points_in_gripper_frame[:, 0] > x_min) & (points_in_gripper_frame[:, 0] < x_max)
        in_y = (points_in_gripper_frame[:, 1] > y_min) & (points_in_gripper_frame[:, 1] < y_max)
        in_z = (points_in_gripper_frame[:, 2] > z_min) & (points_in_gripper_frame[:, 2] < z_max)
        
        count = np.sum(in_x & in_y & in_z)
        scores.append(float(count))

    # Sort grasps by score in descending order
    sorted_grasps_scores = sorted(zip(grasps, scores), key=lambda x: x[1], reverse=True)
    
    if not sorted_grasps_scores:
        return [], []
        
    sorted_grasps, sorted_scores = zip(*sorted_grasps_scores)
    
    return list(sorted_grasps), list(sorted_scores)


def visualize_preprocessing(scene_points, obj_points):
    """
    Visualize the result of the preprocessing step.
    显示预处理的结果：灰色为场景，蓝色为识别出的物体。
    """
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray for scene

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue for object

    print("--> Visualizing scene (grey) and extracted object (blue). Close the window to continue.")
    o3d.visualization.draw_geometries([scene_pcd, obj_pcd])

def visualize_sampled_grasps(obj_points, grasps):
    """
    Visualize the object and all the sampled grasp poses.
    显示物体和所有采样出的抓取姿态。
    """
    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue for object

    geometries = [obj_pcd]

    # Create coordinate frames for all grasps
    for grasp_pose in grasps:
        # Use a smaller size for the coordinate frames to avoid clutter
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        mesh_frame.transform(grasp_pose)
        geometries.append(mesh_frame)
    
    print(f"--> Visualizing object and all {len(grasps)} sampled grasps. Close the window to continue.")
    o3d.visualization.draw_geometries(geometries)


def visualize_filtered_grasps(scene_points, obj_points, valid_grasps, invalid_grasps):
    """
    Visualize the result of grasp filtering.
    显示经过视角筛选后的抓取：坐标系为有效抓取，红色小球为无效抓取。
    """
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue

    geometries = [scene_pcd, obj_pcd]

    # Valid grasps (coordinate frames)
    for grasp_pose in valid_grasps:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        mesh_frame.transform(grasp_pose)
        geometries.append(mesh_frame)
    
    # Invalid grasps (red spheres at origin)
    for grasp_pose in invalid_grasps:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        # The default sphere is at (0,0,0), so we just need to translate it
        sphere.translate(grasp_pose[:3, 3])
        sphere.paint_uniform_color([1, 0, 0]) # Red
        geometries.append(sphere)

    print(f"--> Visualizing filtered grasps: {len(valid_grasps)} valid (frames) and {len(invalid_grasps)} invalid (red spheres).")
    o3d.visualization.draw_geometries(geometries)


def visualize_collision_check_result(scene_points, obj_points, all_checked_grasps, collision_free_grasps):
    """
    Visualize the result of the collision check.
    显示碰撞检测的结果：坐标系为无碰撞的抓取，红色小球为有碰撞的抓取。
    """
    # Find the grasps that were removed due to collision
    collision_free_set = {tuple(g.flatten()) for g in collision_free_grasps}
    collided_grasps = [g for g in all_checked_grasps if tuple(g.flatten()) not in collision_free_set]

    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    scene_pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Gray

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue

    geometries = [scene_pcd, obj_pcd]

    # Collision-free grasps (coordinate frames)
    for grasp_pose in collision_free_grasps:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        mesh_frame.transform(grasp_pose)
        geometries.append(mesh_frame)
    
    # Collided grasps (red spheres at origin)
    for grasp_pose in collided_grasps:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.translate(grasp_pose[:3, 3])
        sphere.paint_uniform_color([1, 0, 0]) # Red
        geometries.append(sphere)

    print(f"--> Visualizing collision check results: {len(collision_free_grasps)} valid (frames) and {len(collided_grasps)} collided (red spheres).")
    o3d.visualization.draw_geometries(geometries)


def visualize_grasps(scene_points, scene_point_colors, obj_points, grasps, scores, gripper_obj_path):
    """
    Visualize the scene, object, and the final top grasps.
    The best grasp is shown with the full gripper mesh.
    显示场景、物体以及最终评分最高的几个抓取。最高分的抓取会显示完整的夹爪模型。
    """
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(scene_points)
    # 使用真实的场景颜色
    if np.max(scene_point_colors) > 1.0:
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_point_colors / 255.0)
    else:
        scene_pcd.colors = o3d.utility.Vector3dVector(scene_point_colors)

    obj_pcd = o3d.geometry.PointCloud()
    obj_pcd.points = o3d.utility.Vector3dVector(obj_points)
    obj_pcd.paint_uniform_color([0, 0.651, 0.929]) # Blue for object

    geometries = [scene_pcd, obj_pcd]

    num_to_show = min(len(grasps), 10)
    
    if num_to_show > 0:
        # Load gripper mesh to show the best grasp
        try:
            gripper_mesh_trimesh = trimesh.load(gripper_obj_path, force='mesh')
            
            # Show best grasp with full mesh
            best_grasp_pose = grasps[0]
            print(best_grasp_pose)
            transformed_gripper = gripper_mesh_trimesh.copy()
            transformed_gripper.apply_transform(best_grasp_pose)
            
            # Convert trimesh to open3d mesh and add to scene
            o3d_gripper_mesh = o3d.geometry.TriangleMesh()
            o3d_gripper_mesh.vertices = o3d.utility.Vector3dVector(transformed_gripper.vertices)
            o3d_gripper_mesh.triangles = o3d.utility.Vector3iVector(transformed_gripper.faces)
            
            o3d_gripper_mesh.compute_vertex_normals()
            o3d_gripper_mesh.paint_uniform_color([0.1, 0.9, 0.1]) # Green for gripper
            geometries.append(o3d_gripper_mesh)

        except Exception as e:
            print(f"Warning: Could not load gripper mesh for visualization: {e}")
            # Fallback to showing the best grasp as a coordinate frame
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            mesh_frame.transform(grasps[0])
            geometries.append(mesh_frame)

        # Show other top grasps as coordinate frames
        for i in range(1, num_to_show):
            grasp_pose = grasps[i]
            mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03) # slightly smaller
            mesh_frame.transform(grasp_pose)
            geometries.append(mesh_frame)
    
    print(f"--> Visualizing scene, object, and top {num_to_show} grasps (best grasp shown with gripper mesh).")
    o3d.visualization.draw_geometries(geometries)


def main():
    # basename = "20250830_220551"
    basename = get_newest(RAW_DATA_DIR)
    scene_points_path = f"{RAW_DATA_DIR}/{basename}/pointcloud_{basename}_points.npy"
    scene_point_color_path = f"{RAW_DATA_DIR}/{basename}/pointcloud_{basename}_colors.npy"
    mask_path = f"{RAW_DATA_DIR}/{basename}/object_mask.png"
    gripper_obj_path= GRIPPER_MODEL_PATH
    sample_num = 500
    min_dist = 0.8      # filter object outliers, min distance from camera in meters
    max_dist = 1.8      # filter object outliers, max distance from camera in meters
    z_offset = 0.12     # z_offset in meters, adjust to get better grasp depth
    angle_thresholds = {    # filter grasps by angle to camera axes
        'x': (0, 90),
        'y': (0, 90),
        'z': (0, 90)
    }

    scene_points = np.load(scene_points_path)
    scene_point_colors = np.load(scene_point_color_path)
    object_mask = Image.open(mask_path)

    # Preprocess the data
    print("1. Preprocessing the data...")
    initial_obj_points = preprocess(scene_points, object_mask)

    # 1.5 Filter object points by distance from camera
    obj_points = filter_object_points_by_distance(
        initial_obj_points, 
        min_dist=min_dist, 
        max_dist=max_dist
    )
    # visualize_preprocessing(scene_points, obj_points)

    # Sample grasp configurations
    print("2. Sampling grasp configurations...")
    grasps = sample_grasp(obj_points, num_grasps=sample_num, z_offset=z_offset)        # z_offset in meters
    # visualize_sampled_grasps(obj_points, grasps)

    print("\n2.5. Filtering grasps by viewpoint...")
    print(f"   Filtering with angle thresholds: {angle_thresholds}")
    filtered_grasps, discarded_grasps = filter_grasps_by_orientation(grasps, angle_thresholds)
    print(f"   Kept {len(filtered_grasps)} grasps facing the viewpoint, discarded {len(discarded_grasps)}.")
    # visualize_filtered_grasps(scene_points, obj_points, filtered_grasps, discarded_grasps)

    # Check for collisions
    print(f"3. Checking for collisions...")
    valid_grasps = check_collision(scene_points, filtered_grasps, gripper_obj_path)
    print(f"   Found {len(filtered_grasps)} potential samples, {len(valid_grasps)} are collision-free.")
    visualize_collision_check_result(scene_points, obj_points, filtered_grasps, valid_grasps)

    print("4. Checking grasp quality...")
    sorted_grasps, scores = check_grasp_quality(valid_grasps, obj_points, gripper_depth=z_offset)

    print("\n--- Top 5 Grasp Results ---")
    for i in range(min(5, len(sorted_grasps))):
        print(f"Grasp {i+1}: Score = {scores[i]}")
        # print(f"  Pose:\n{sorted_grasps[i]}")

    save_path = f"{RAW_DATA_DIR}/{basename}/grasps.npz"
    if sorted_grasps:
        np.savez(save_path, grasps=np.array(sorted_grasps), scores=np.array(scores))
        print(f"\nSaved {len(sorted_grasps)} grasps and their scores to {save_path}")
    else:
        print("\nNo valid grasps found to save.")

    print("\n5. Visualizing results...")
    if obj_points.shape[0] > 0 and sorted_grasps:
        visualize_grasps(scene_points, scene_point_colors, obj_points, sorted_grasps, scores, gripper_obj_path)
    else:
        print("Skipping visualization because no valid grasps were found.")

if __name__ == "__main__":
    main()