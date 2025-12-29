import torch

def multiview_fusion(Rt_list: list[torch.Tensor], P_list: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Multiview fusion given camera extrinsics and point clouds
    Args:
        Rt_list (list[torch.Tensor]): list of camera extrinsics, each element is a 4x4 matrix representing the camera transformation from camera to world
        P_list (list[torch.Tensor]): list of point clouds, each element is a N_ix3 matrix representing the 3D points in the camera coordinate system

    Returns:
        list[torch.Tensor]: list of fused point clouds(every one in world coordinate)
    """
    
    assert len(Rt_list) == len(P_list), "The same number of cameras and point clouds are required"
    
    # Initialize the list to store the fused point clouds
    fused_point_clouds = []
    
    for Rt, P in zip(Rt_list, P_list):
        # Check if the point cloud is empty
        assert P.shape[0] > 0, "Point cloud is empty"
        
        # Transform the point cloud to world coordinates using the camera extrinsics
        ones = torch.ones(P.shape[0], 1, device=P.device)
        P_homo = torch.cat([P, ones], dim=1) #[N, 4]
        P_transformed = (Rt @ P_homo[...,None])[...,0] #[N, 4]
        fused_point_clouds.append(P_transformed[:, :3])
        
    return fused_point_clouds