import numpy as np 
from rel import DATA_PATH, XARM7_URDF_PATH
from rel.robots.pk_robot import XArm7

from pov.datasets.pov_zarr_dataset import PovZarrDataset
from pov.utils.numpy.common import preallocate_and_concatenate

def get_current_robot_points(concatenated_canonical_points, concatenated_point_body_indices, current_transforms):
    canonical_points = concatenated_canonical_points  # n_all_pc, 3
    point_body_indices = concatenated_point_body_indices  # n_all_pc,

    current_rotations = current_transforms[:, :3, :3]  # n_body, 3, 3
    current_translations = current_transforms[:, :3, 3]  # n_body, 3
    current_point_rotations = current_rotations[
        point_body_indices
    ]  # n_all_pc, 3, 3
    current_point_translations = current_translations[
        point_body_indices
    ]  # n_all_pc, 3
    canonical_points_expanded = canonical_points[:, :, np.newaxis]  # n_all_pc, 3, 1
    current_robot_points = (
        np.matmul(current_point_rotations, canonical_points_expanded).squeeze(-1)
        + current_point_translations
    )  # n_all_pc, 3
    return current_robot_points

def get_current_robot_points_batch(concatenated_canonical_points, concatenated_point_body_indices, current_transforms):
    """Batched version of get_current_robot_points.
    
    Args:
        concatenated_canonical_points: (n_all_pc, 3) canonical points
        concatenated_point_body_indices: (n_all_pc,) body indices for each point
        current_transforms: (b, n_body, 4, 4) current transformation matrices
        
    Returns:
        current_robot_points: (b, n_all_pc, 3) transformed points in current pose
    """
    # Extract rotations and translations from transforms
    current_rotations = current_transforms[:, :, :3, :3]  # b, n_body, 3, 3
    current_translations = current_transforms[:, :, :3, 3]  # b, n_body, 3
    
    # Gather the rotations and translations for each point based on body indices
    current_point_rotations = current_rotations[:, concatenated_point_body_indices]  # b, n_all_pc, 3, 3
    current_point_translations = current_translations[:, concatenated_point_body_indices]  # b, n_all_pc, 3
    
    # Expand canonical points for matrix multiplication
    canonical_points_expanded = concatenated_canonical_points[:, :, np.newaxis]  # n_all_pc, 3, 1
    
    # Add batch dimension to canonical points to enable broadcasting
    canonical_points_expanded = canonical_points_expanded[np.newaxis]  # 1, n_all_pc, 3, 1
    
    # Transform points
    current_robot_points = (
        np.matmul(current_point_rotations, canonical_points_expanded).squeeze(-1)  # b, n_all_pc, 3
        + current_point_translations  # b, n_all_pc, 3
    )
    return current_robot_points


if __name__ == "__main__":
    ##########################################################################################
    # Hyperparameters
    ##########################################################################################
    exp_dir = DATA_PATH / "0428_9points_zixuan" / "raw_data"
    start_idx = 9
    end_idx = 17
    save_dir = exp_dir.parent / f"{start_idx}_{end_idx}_rpt.zarr"

    ##########################################################################################
    # Main code
    ##########################################################################################
    arm = XArm7(device="cuda")
    body_pc_downsampled_path = XARM7_URDF_PATH.parent / "body_pc_downsampled"
    body_name_list_path = body_pc_downsampled_path / "body_name_list.npy"
    body_name_to_pc_canonical_downsampled_path = body_pc_downsampled_path / "body_name_to_pc_canonical_downsampled.npy"
    concatenated_canonical_points_path = body_pc_downsampled_path / "concatenated_canonical_points.npy"
    concatenated_point_body_indices_path = body_pc_downsampled_path / "concatenated_point_body_indices.npy"
    body_name_list = np.load(body_name_list_path, allow_pickle=True)
    body_name_to_pc_canonical_downsampled = np.load(body_name_to_pc_canonical_downsampled_path, allow_pickle=True)
    concatenated_canonical_points = np.load(concatenated_canonical_points_path, allow_pickle=True)
    concatenated_point_body_indices = np.load(concatenated_point_body_indices_path, allow_pickle=True)

    robot0_all_qpos = []
    current_robot_points_arrays = []
    target_robot_points_arrays = []
    point_clouds_arrays = []
    original_actions_arrays = []
    episode_ends_arrays = []

    pov_zarr_dataset = PovZarrDataset(
        save_path=str(save_dir),
    )

    reference_open_joint_values = arm.reference_joint_values
    reference_close_joint_values = reference_open_joint_values.clone()
    reference_close_joint_values[-6:] = 0.85

    reference_open_status = arm.pk_chain.forward_kinematics(
        th=arm.ensure_tensor(reference_open_joint_values)
    )
    X_WorldGripper = reference_open_status["xarm_gripper_base_link"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)
    X_WorldTcp = reference_open_status["link_tcp"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)
    X_WorldLfopen = reference_open_status["left_finger"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)
    X_WorldRfopen = reference_open_status["right_finger"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)

    X_TcpGripper = np.linalg.inv(X_WorldTcp) @ X_WorldGripper  # (4, 4)
    X_GripperLfopen = np.linalg.inv(X_WorldGripper) @ X_WorldLfopen  # (4, 4)
    X_GripperRfopen = np.linalg.inv(X_WorldGripper) @ X_WorldRfopen  # (4, 4)

    reference_close_status = arm.pk_chain.forward_kinematics(
        th=arm.ensure_tensor(reference_close_joint_values)
    )
    X_WorldGripper2 = reference_close_status["xarm_gripper_base_link"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)
    X_WorldLfclose = reference_close_status["left_finger"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)
    X_WorldRfclose = reference_close_status["right_finger"].get_matrix().detach().cpu().numpy()[0]  # (4, 4)

    X_GripperLfclose = np.linalg.inv(X_WorldGripper2) @ X_WorldLfclose  # (4, 4)
    X_GripperRfclose = np.linalg.inv(X_WorldGripper2) @ X_WorldRfclose  # (4, 4)
    
    total_step_count = 0
    for i in range(start_idx, end_idx + 1):
        file_path = exp_dir / f"{i}.npz"
        data = np.load(file_path, allow_pickle=True)

        robot_joint_values = data["proprioceptions"] # (n, 7+1)
        robot_joint_values[:, -1] /= 1000.0  # 850 -> 0.85
        robot_joint_values[:, -1] = 0.85 - robot_joint_values[:, -1]
        robot_joint_values_w_mimic = np.concatenate([robot_joint_values, np.repeat(robot_joint_values[:, -1:], 5, axis=1)], axis=1)  # (n, 7+1+1*5=13)
        current_status = arm.pk_chain.forward_kinematics(
            th=arm.ensure_tensor(robot_joint_values_w_mimic)
        )

        current_transforms = []
        for link_name in body_name_list:
            current_transforms.append(current_status[link_name].get_matrix().detach().cpu().numpy())  # [(b, 4, 4), ...]
        current_transforms = np.stack(current_transforms, axis=1)  # (b, n_body, 4, 4)
        current_robot_points = get_current_robot_points_batch(concatenated_canonical_points, concatenated_point_body_indices, current_transforms)  # (b, n_all_pc, 3)
        
        # get target robot points
        b = data["original_actions"].shape[0]
        pos_WorldEENext = data["original_actions"][:, :3]  # (b, 3)
        R_WorldEENext = data["original_actions"][:, 3:12].reshape(b, 3, 3)  # (b, 3, 3)
        X_WorldTcptgt = np.zeros((b, 4, 4))
        X_WorldTcptgt[:, 3, 3] = 1.0
        X_WorldTcptgt[:, :3, :3] = R_WorldEENext
        X_WorldTcptgt[:, :3, 3] = pos_WorldEENext
        X_WorldGrippertgt = np.matmul(X_WorldTcptgt, X_TcpGripper[np.newaxis, ...].repeat(b, axis=0))  # (b, 4, 4)
        X_WorldLfopentgt = np.matmul(X_WorldGrippertgt, X_GripperLfopen[np.newaxis, ...].repeat(b, axis=0))  # (b, 4, 4)
        X_WorldRfopentgt = np.matmul(X_WorldGrippertgt, X_GripperRfopen[np.newaxis, ...].repeat(b, axis=0))  # (b, 4, 4)
        X_WorldLfclosetgt = np.matmul(X_WorldGrippertgt, X_GripperLfclose[np.newaxis, ...].repeat(b, axis=0))  # (b, 4, 4)
        X_WorldRfclosetgt = np.matmul(X_WorldGrippertgt, X_GripperRfclose[np.newaxis, ...].repeat(b, axis=0))  # (b, 4, 4)
        gripper_is_close = data["original_actions"][:, -1] < 425  # (b,)
        X_WorldLftgt = np.where(gripper_is_close[:, np.newaxis, np.newaxis], X_WorldLfclosetgt, X_WorldLfopentgt)  # (b, 4, 4)
        X_WorldRftgt = np.where(gripper_is_close[:, np.newaxis, np.newaxis], X_WorldRfclosetgt, X_WorldRfopentgt)  # (b, 4, 4)
        target_transforms = np.stack([X_WorldGrippertgt, X_WorldLftgt, X_WorldRftgt], axis=1)  # (b, 3, 4, 4)
        target_robot_points = get_current_robot_points_batch(concatenated_canonical_points, concatenated_point_body_indices, target_transforms)  # (b, n_all_pc, 3)

        current_robot_points_arrays.append(current_robot_points)
        target_robot_points_arrays.append(target_robot_points)
        robot0_all_qpos.append(data["proprioceptions"])
        point_clouds_arrays.append(data["pointclouds"])
        original_actions_arrays.append(data["original_actions"])

        total_step_count += len(data["proprioceptions"])
        episode_ends_arrays.append(total_step_count)

    current_robot_points_arrays = preallocate_and_concatenate(current_robot_points_arrays, axis=0)
    target_robot_points_arrays = preallocate_and_concatenate(target_robot_points_arrays, axis=0)
    original_actions_arrays = preallocate_and_concatenate(original_actions_arrays, axis=0)
    point_clouds_arrays = preallocate_and_concatenate(point_clouds_arrays, axis=0)
    robot0_all_qpos = preallocate_and_concatenate(robot0_all_qpos, axis=0)


    pov_zarr_dataset.save_data({
        "data/obs/current_robot_points": np.asarray(np.transpose(current_robot_points_arrays, (0, 2, 1)), dtype=np.float32),
        "data/actions/target_robot_points": np.asarray(np.transpose(target_robot_points_arrays, (0, 2, 1)), dtype=np.float32),
        "data/obs/robot0_all_qpos": np.asarray(robot0_all_qpos, dtype=np.float32),
        "data/obs/point_clouds": np.asarray(np.transpose(point_clouds_arrays, (0, 2, 1)), dtype=np.float32),
        "data/actions/original_actions": np.asarray(original_actions_arrays, dtype=np.float32),
        "meta/episode_ends": np.asarray(episode_ends_arrays, dtype=np.int64),
    })

    pov_zarr_dataset.print_structure()
