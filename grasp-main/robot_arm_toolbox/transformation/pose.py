import numpy as np
from scipy.spatial.transform import Rotation as R

def translation_rotation_2_matrix(translation, rotation):
    '''
    **Input:**

    - translation: numpy array of shape (3,)

    - rotation: numpy array of shape (3,3)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    return np.vstack((np.hstack((rotation, translation.reshape((3,1)))), np.array((0,0,0,1.0))))

def matrix_2_translation_rotation(matrix):
    '''
    **Input:**

    - Homogeneous transformation matrix of shape (4,4).

    **Output:**

    - translation: numpy array of shape (3,)

    - rotation: numpy array of shape (3,3)
    '''
    translation = matrix[:3,3]
    rotation = matrix[:3,:3]
    return translation, rotation

def translation_rotation_2_array(translation, rotation):
    '''
    **Input:**

    - translation: numpy array of shape (3,)

    - rotation: numpy array of shape (3,3)
    
    **Output:**

    - pose array of shape (6,).
    '''
    rotation_vector = R.from_matrix(rotation).as_rotvec()
    # print(f't:{translation}, r:{rotation_vector}')
    return np.concatenate((translation, rotation_vector))

def pose_array_2_matrix(pose):
    '''
    **Input:**

    - pose: numpy array of shape (6,)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    pose = pose.astype(np.float32)
    translation = pose[:3] 
    rotation = R.from_rotvec(pose[3:]).as_matrix()
    return translation_rotation_2_matrix(translation, rotation)
    
def pose_matrix_2_array(matrix):
    '''
    **Input:**

    - matrix: Homogeneous transformation matrix of shape (4,4).
    
    **Output:**

    - pose: numpy array of shape (6,)
    '''
    translation = matrix[:3,3].reshape(3)
    rotation_matrix = matrix[:3,:3]
    rotation = R.from_matrix(rotation_matrix).as_rotvec()
    return np.concatenate((translation, rotation))

def translation_euler_2_matrix(pose):
    '''
    **Input:**

    - pose: numpy array of shape (6,)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    pose = pose.astype(np.float32)
    translation = pose[:3] 
    rotation = R.from_euler('xyz', pose[3:]).as_matrix()
    return translation_rotation_2_matrix(translation, rotation)

def matrix_2_translation_euler(matrix):
    '''
    **Input:**

    - pose: numpy array of shape (6,)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    matrix = matrix.astype(np.float32)
    translation = matrix[:3, 3] 
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz')
    return np.concatenate((translation, rotation))

def xarm_translation_euler_2_matrix(pose):
    '''
    **Input:**

    - pose: numpy array of shape (6,)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    pose = pose.astype(np.float32)
    translation = pose[:3]  / 1000
    rotation = R.from_euler('xyz', pose[3:], degrees=True).as_matrix()
    return translation_rotation_2_matrix(translation, rotation)

def matrix_2_xarm_translation_euler(matrix):
    '''
    **Input:**

    - pose: numpy array of shape (6,)
    
    **Output:**

    - Homogeneous transformation matrix of shape (4,4).
    '''
    matrix = matrix.astype(np.float32)
    translation = matrix[:3, 3] * 1000
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz')/np.pi*180
    return np.concatenate((translation, rotation))