import trimesh
import viser
from diff_robot_hand.utils.sample_utils import sample_hand_init_poses_with_face_retreat_distances
from diff_robot_hand.hand_model import XArmGripper
from diff_robot_hand.utils.mesh_and_urdf_utils import as_mesh
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import os 
from pathlib import Path

object_mesh_dir_path = Path("/home/aris/projects/differentiable_robot_hand/data/object_mesh")
# ----------------- Load object mesh -----------------
# - object_mesh 
#   - apple
#       - apple.obj
#  - banana
#      - banana.obj    
# - ...
# ----------------------------------------------------
# Recursively search for all .obj files in the directory
object_mesh_path_list = sorted(list(object_mesh_dir_path.rglob("*.obj")))
# divide the object meshes into two groups: convex and non-convex based on the file name: whether it contains "coacd"

coacd_mesh_path_list = sorted([path for path in object_mesh_path_list if "coacd" in path.stem])
original_mesh_path_list = sorted([path for path in object_mesh_path_list if "coacd" not in path.stem])
# Load the meshes
test_object_trimeshes = [trimesh.load_mesh(str(mesh_path)) for mesh_path in original_mesh_path_list]
test_object_coacd_trimeshes = [trimesh.load_mesh(str(mesh_path)) for mesh_path in coacd_mesh_path_list]
assert len(test_object_trimeshes) == len(test_object_coacd_trimeshes)

# test_object_trimeshes = [
#     trimesh.creation.box((0.075, 0.075, 0.075)),
#     trimesh.creation.capsule(0.05, 0.03),
#     trimesh.creation.cone(0.04, 0.08),
#     trimesh.creation.cylinder(0.03, 0.05),
#     trimesh.creation.uv_sphere(0.04, (12, 12)),
# ]

def in_collision_with_gripper(object_mesh, gripper_meshes, gripper_transforms, silent=False):
    """Check collision of object with gripper.

    Arguments:
        object_mesh {trimesh} -- mesh of object
        gripper_transforms {list of numpy.array} -- homogeneous matrices of gripper
        gripper_name {str} -- name of gripper

    Keyword Arguments:
        silent {bool} -- verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object mesh
    """
    manager = trimesh.collision.CollisionManager()
    manager.add_object('object', object_mesh)
    min_distance = []
    for tf in tqdm(gripper_transforms, disable=silent):
        # min_distance.append(manager.min_distance_single(gripper_mesh, transform=tf))
        min_distance.append(np.min([manager.min_distance_single(
                    gripper_mesh, transform=tf) for gripper_mesh in gripper_meshes]))
    return [d < 0 for d in min_distance], min_distance

def in_collision_with_gripper_coacd(object_meshes, gripper_meshes, gripper_transforms, silent=False):
    """Check collision of object with gripper for convex parts.

    Arguments:
        object_meshes {list of trimesh.Trimesh} -- Convex part meshes of an object
        gripper_meshes {list of trimesh.Trimesh} -- Meshes of the gripper
        gripper_transforms {list of numpy.array} -- Homogeneous matrices of gripper transforms

    Keyword Arguments:
        silent {bool} -- Verbosity (default: {False})

    Returns:
        [list of bool] -- Which gripper poses are in collision with object meshes
    """
    # Initialize collision managers for object meshes and gripper meshes
    object_manager = trimesh.collision.CollisionManager()
    gripper_manager = trimesh.collision.CollisionManager()
    
    # Add object meshes to the object manager
    for i, obj_mesh in enumerate(object_meshes):
        object_manager.add_object(f'object_part_{i}', obj_mesh)
    
    # Prepare to store the results
    collision_results = []

    # Check collisions for each gripper pose
    for tf in tqdm(gripper_transforms, disable=silent):
        # Clear the gripper manager and add gripper meshes with their transformed poses
        gripper_manager = trimesh.collision.CollisionManager()  # reset for each pose
        for i, gripper_mesh in enumerate(gripper_meshes):
            gripper_manager.add_object(f'gripper_part_{i}', gripper_mesh, transform=tf)

        # Check if any object part collides with the gripper
        is_collision = object_manager.in_collision_other(gripper_manager)
        collision_results.append(is_collision)

    return collision_results


def grasp_quality_antipodal(transforms, object_mesh, left_ray_origins, left_ray_directions, 
                            right_ray_origins, right_ray_directions, 
                            gripper_width, silent=False):
    """Grasp quality function.

    Arguments:
        transforms {numpy.array} -- grasps
        collisions {list of bool} -- collision information
        object_mesh {trimesh} -- object mesh

    Keyword Arguments:
        gripper_name {str} -- name of gripper (default: {'panda'})
        silent {bool} -- verbosity (default: {False})

    Returns:
        list of float -- quality of grasps [0..1]
    """
    res = []
    refined_p_list = []
    intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(
        object_mesh, scale_to_box=True)
    
    for p in tqdm(transforms, total=len(transforms), disable=silent):
        refined_p = p.copy()
        n_left_rays = left_ray_origins.shape[0]
        n_right_rays = right_ray_origins.shape[0]
        ray_origins = np.concatenate([left_ray_origins, right_ray_origins], axis=0)  # (n_rays, 3)
        ray_directions = np.concatenate([left_ray_directions, right_ray_directions], axis=0)  # (n_rays, 3)

        # transform ray origins 
        ray_origins = np.dot(p[:3, :3], ray_origins.T).T + p[:3, 3]  # (n_rays, 3)
        ray_directions = np.dot(p[:3, :3], ray_directions.T).T  # (n_rays, 3)   

        # left rays (n_left_rays, ), (21, )
        # right rays (n_right_rays, ) 
        # --> hit 
        # locations (n_rays_hit, 3)
        # index_ray (n_rays_hit, ), index in ray_origins
        # index_tri (n_rays_hit, ), index in object_mesh.faces
        # --> divide into left and right rays
        # index_ray_left (n_left_rays_hit, ), index in ray_origins
        # index_ray_right (n_right_rays_hit, ), index in ray_origins
        # --> tell whether the ray is hitting the object
        # --> if hitting, tell which contact point is closest to the finger (which would be hit first during closing)


        locations, index_ray, index_tri = intersector.intersects_location(
            ray_origins, ray_directions, multiple_hits=False)
        
        if locations.size == 0:
            res.append(0)
            refined_p_list.append(refined_p)
        else:
            ray_dis = np.linalg.norm(ray_origins[index_ray] - locations, axis=1)  # (n_rays_hit, )
            valid_ray_mask = ray_dis < gripper_width
            valid_ray_idx = index_ray[valid_ray_mask]  # index in ray_origins  , (18, )
            valid_locations = locations[valid_ray_mask]  # (18, 3)
            valid_index_tri = index_tri[valid_ray_mask]  # (18, )

            if valid_ray_idx.size == 0:
                res.append(0)
                refined_p_list.append(refined_p)
            else:
                # select the contact point closest to the finger (which would be hit first during closing)
                index_ray_left_valid = valid_ray_idx[valid_ray_idx < n_left_rays]  # (n_left_valid_rays_hit, ), global index
                index_ray_right_valid = valid_ray_idx[valid_ray_idx >= n_left_rays]  # (n_right_valid_rays_hit, ), global index
                
                if index_ray_left_valid.size == 0 or index_ray_right_valid.size == 0:
                    res.append(0)
                    refined_p_list.append(refined_p)
                    continue

                left_contact_idx_in_left_valid_hit = np.linalg.norm(
                    ray_origins[index_ray_left_valid] - valid_locations[:len(index_ray_left_valid)], axis=1).argmin()    # local index

                right_contact_idx_in_right_valid_hit = np.linalg.norm(
                    ray_origins[index_ray_right_valid] - valid_locations[len(index_ray_left_valid):], axis=1).argmin()  # local index
                
                left_ray_min_origins = ray_origins[index_ray_left_valid[left_contact_idx_in_left_valid_hit]]
                right_ray_min_origins = ray_origins[index_ray_right_valid[right_contact_idx_in_right_valid_hit]]
                mid_ray_min_origins = (left_ray_min_origins + right_ray_min_origins) / 2

                left_contact_point = valid_locations[left_contact_idx_in_left_valid_hit]
                right_contact_point = valid_locations[len(index_ray_left_valid) + right_contact_idx_in_right_valid_hit]
                mid_contact_point = (left_contact_point + right_contact_point) / 2
                refine_delta_translate = mid_contact_point - mid_ray_min_origins
                refined_p[:3, 3] += refine_delta_translate

                left_contact_normal = object_mesh.face_normals[valid_index_tri[left_contact_idx_in_left_valid_hit]]
                right_contact_normal = object_mesh.face_normals[valid_index_tri[len(index_ray_left_valid) + right_contact_idx_in_right_valid_hit]]

                l_to_r = (right_contact_point - left_contact_point) / \
                    np.linalg.norm(right_contact_point -
                                    left_contact_point)
                r_to_l = (left_contact_point - right_contact_point) / \
                    np.linalg.norm(left_contact_point -
                                    right_contact_point)

                qual_left = np.dot(left_contact_normal, r_to_l)
                qual_right = np.dot(right_contact_normal, l_to_r)
                if qual_left < 0 or qual_right < 0:
                    qual = 0
                else:
                    qual = min(qual_left, qual_right)

                res.append(qual)
                refined_p_list.append(refined_p)
    return res, refined_p_list


if __name__ == "__main__":
    object_id = 8
    device = "cuda:0"
    server = viser.ViserServer()
    hand = XArmGripper(device=device)
    hand_open_mesh:trimesh.Scene = hand.get_hand_trimesh(np.zeros(hand.ndof), visual=False, collision=True)["collision"]
    hand_open_meshes = hand_open_mesh.dump()
    # params
    n_sample_object_points = 2048
    n_sample_init_hand_posistions = 512
    n_sample_init_hand_rotations = 5
    n_sample_face_retreat_distances = 3
    face_retreat_distance_range = (0.005, 0.02)
    gripper_width = 0.08

    ori_face_vector=np.array([0.0, 0.0, 1.0])
    palm_face_center_offset=np.array([0.0, 0.0, 0.12])  # 0.12 -> 0.16
    right_finger_ray_origins = np.array(
        [[ 0.0, -4.44753767e-02,  1.27332952e-01],
        [ 0.0, -4.44753767e-02,  1.31856788e-01], 
        [ 0.0, -4.44753767e-02,  1.36380624e-01],
        [ 0.0, -4.44753767e-02,  1.40904460e-01],
        [ 0.0, -4.44753767e-02,  1.45428296e-01],
        [ 0.0, -4.44753767e-02,  1.49952132e-01],
        [ 0.0, -4.44753767e-02,  1.54475968e-01],
        [0.008, -4.44753767e-02,  1.27332952e-01],
        [0.008, -4.44753767e-02,  1.31856788e-01],
        [0.008, -4.44753767e-02,  1.36380624e-01],
        [0.008, -4.44753767e-02,  1.40904460e-01],
        [0.008, -4.44753767e-02,  1.45428296e-01],
        [0.008, -4.44753767e-02,  1.49952132e-01],
        [0.008, -4.44753767e-02,  1.54475968e-01],
        [-0.008, -4.44753767e-02,  1.27332952e-01],
        [-0.008, -4.44753767e-02,  1.31856788e-01],
        [-0.008, -4.44753767e-02,  1.36380624e-01],
        [-0.008, -4.44753767e-02,  1.40904460e-01],
        [-0.008, -4.44753767e-02,  1.45428296e-01],
        [-0.008, -4.44753767e-02,  1.49952132e-01],
        [-0.008, -4.44753767e-02,  1.54475968e-01],]
    )
    right_finger_ray_directions = np.array(
        [[0.0, 1.0, 0.0] for _ in range(right_finger_ray_origins.shape[0])]
    )
    left_finger_ray_origins = np.array(
        [[0.0, 4.44753767e-02,  1.27332952e-01],
        [0.0, 4.44753767e-02,  1.31856788e-01], 
        [0.0, 4.44753767e-02,  1.36380624e-01],
        [0.0, 4.44753767e-02,  1.40904460e-01],
        [0.0, 4.44753767e-02,  1.45428296e-01],
        [0.0, 4.44753767e-02,  1.49952132e-01],
        [0.0, 4.44753767e-02,  1.54475968e-01],
        [0.008, 4.44753767e-02,  1.27332952e-01],
        [0.008, 4.44753767e-02,  1.31856788e-01],
        [0.008, 4.44753767e-02,  1.36380624e-01],
        [0.008, 4.44753767e-02,  1.40904460e-01],
        [0.008, 4.44753767e-02,  1.45428296e-01],
        [0.008, 4.44753767e-02,  1.49952132e-01],
        [0.008, 4.44753767e-02,  1.54475968e-01],
        [-0.008, 4.44753767e-02,  1.27332952e-01],
        [-0.008, 4.44753767e-02,  1.31856788e-01],
        [-0.008, 4.44753767e-02,  1.36380624e-01],
        [-0.008, 4.44753767e-02,  1.40904460e-01],
        [-0.008, 4.44753767e-02,  1.45428296e-01],
        [-0.008, 4.44753767e-02,  1.49952132e-01],
        [-0.008, 4.44753767e-02,  1.54475968e-01],
    ])
    left_finger_ray_directions = np.array(
        [[0.0, -1.0, 0.0] for _ in range(left_finger_ray_origins.shape[0])]
    )

    for object_id in range(len(test_object_trimeshes)):
        print(f"---{object_id}---")
        object_path = original_mesh_path_list[object_id] 
        object_coacd_path = coacd_mesh_path_list[object_id]
        print(object_path)
        print(object_coacd_path)

        object_mesh = test_object_trimeshes[object_id]
        object_coacd_mesh = test_object_coacd_trimeshes[object_id]

        object_pc, object_face_idx = trimesh.sample.sample_surface(
            object_mesh, n_sample_object_points
        )
        object_normals = object_mesh.face_normals[object_face_idx]

        hand_poses = sample_hand_init_poses_with_face_retreat_distances(
            object_pc=object_pc,
            object_normals=object_normals,
            palm_face_center_offset=palm_face_center_offset,
            ori_face_vector=ori_face_vector,
            face_retreat_distance_range=face_retreat_distance_range,
            n_sample_init_hand_positions=n_sample_init_hand_posistions,
            n_sample_init_hand_rotations=n_sample_init_hand_rotations,
            n_sample_face_retreat_distances=n_sample_face_retreat_distances,
        )
        print(hand_poses.shape)
        gripper_open_mesh = hand
        object_coacd_mesh_list = object_coacd_mesh.split()

        hand_pose_in_collision = in_collision_with_gripper_coacd(object_coacd_mesh_list, hand_open_meshes, hand_poses, silent=True)
        valid_idx = np.where(np.array(hand_pose_in_collision) == False)[0]
        valid_hand_poses = hand_poses[valid_idx]
        print(len(valid_idx))

        server.scene.add_mesh_trimesh("object_mesh", object_mesh)


        qual, refined_p_list = grasp_quality_antipodal(valid_hand_poses, object_mesh, left_finger_ray_origins, 
                                    left_finger_ray_directions, 
                                    right_finger_ray_origins, right_finger_ray_directions, 
                                    gripper_width, silent=True)
        cmap = plt.get_cmap("rainbow")
        qual_colors = cmap(np.array(qual))[:, :3]
        # print(qual)
        print(len(qual))
        print(len(np.where(np.array(qual) > 0.9)[0]))
        print(len(np.where(np.array(qual) > 0.8)[0]))

        # save the results
        object_mesh_path = original_mesh_path_list[object_id]
        object_grasp_save_path = object_mesh_path.parent / "grasp_new.pkl"
        print(object_grasp_save_path)
        import pickle
        qual_is_good = np.array(qual) > 0.85
        refined_p_list = np.array(refined_p_list)
        refined_p_is_good = refined_p_list[qual_is_good]
        with open(object_grasp_save_path, "wb") as f:
            pickle.dump(refined_p_is_good, f)


        # for i, valid_hand_pose in enumerate(refined_p_list):
        #     if qual[i] > 0.85:
        #         transform_matrix = valid_hand_pose
        #         pos = transform_matrix[:3, 3]
        #         wxyz = trimesh.transformations.quaternion_from_matrix(transform_matrix)
        #         hand_open_mesh = as_mesh(hand_open_mesh)
        #         hand_open_mesh.visual.face_colors = qual_colors[i]
        #         server.scene.add_mesh_trimesh(f"hand_{i}_collision", hand_open_mesh, wxyz=wxyz, position=pos, visible=False)

        # server.scene.add_mesh_trimesh("object_mesh", object_mesh)
        # import time 
        # while True:
        #     time.sleep(1)
    