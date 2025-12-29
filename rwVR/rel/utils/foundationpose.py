from FoundationPose.estimater import *

    
def single_frame_registration(mesh, K, rgb, depth, mask):
    """
    Input: 
    - mesh: trimesh.Trimesh
    - K: np.ndarray, camera intrinsic matrix | (3, 3) 
            e.g.array([[319.58200073,   0.        , 320.21498477],
                        [  0.        , 417.11868286, 244.34866809],
                        [  0.        ,   0.        ,   1.        ]])
    - rgb: np.ndarray, rgb image | (h, w, 3), uint8
    - depth: np.ndarray, depth image, in meters | (h, w), float64
    - mask: np.ndarray, binary mask of the object  | (h, w), bool

    Output:
    - X_CamObj: np.ndarray, camera to object transformation matrix | (4, 4)
    """
    est = FoundationPose(
        model_pts=mesh.vertices, 
        model_normals=mesh.vertex_normals, 
        mesh=mesh, 
    )
    X_CamObj = est.register(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
            
    return X_CamObj
