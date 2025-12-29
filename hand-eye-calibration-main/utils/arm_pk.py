"""
Robot arm model base class using PyTorch Kinematics (arm_p(ytorch)k(inematics))
"""

import torch
import trimesh
import numpy as np
from pathlib import Path
import pytorch_kinematics as pk
from typing import Optional, Union, Dict


class RobotArmPK:
    """Base class for robotic arm models providing core functionality.
    
    Attributes:
        urdf_path (Path): Path to the URDF file
        actuated_joint_names (list): Names of actuated joints
        lower_joint_limits_np (np.ndarray): Lower joint limits in numpy array
        upper_joint_limits_np (np.ndarray): Upper joint limits in numpy array
        reference_joint_values_np (np.ndarray): Reference joint values in numpy array
    """
    
    def __init__(
        self,
        urdf_path: Path,
        load_visual_mesh: bool = True,
        load_col_mesh: bool = True,
        dtype=torch.float,
        device="cuda",
    ):
        """Initialize the robotic arm model.
        
        Args:
            urdf_path: Path to the URDF file
            load_visual_mesh: Whether to load visual meshes
            load_col_mesh: Whether to load collision meshes
            dtype: Data type for tensors
            device: Device to use (cuda/cpu)
        """
        self.urdf_path = urdf_path
        self.dtype = dtype
        self.device = device
        
        # Initialize kinematics chain
        self._init_kinematics_chain()
        
        # Load meshes if requested
        if load_visual_mesh or load_col_mesh:
            self._load_meshes(load_visual_mesh, load_col_mesh)
        
        # Set reference joint values (can be overridden by child classes)
        self.set_reference_joint_values()

    def _init_kinematics_chain(self):
        """Initialize the kinematics chain from URDF."""
        self.pk_chain = pk.build_chain_from_urdf(
            open(str(self.urdf_path)).read().encode()
        )
        self.pk_chain.to(device=self.device, dtype=self.dtype)
        
        # Get joint information
        self.actuated_joint_names = self.pk_chain.get_joint_parameter_names(exclude_fixed=True)
        self.ndof = len(self.actuated_joint_names)
        
        # Get joint limits
        lower_limits, upper_limits = self.pk_chain.get_joint_limits()
        self.lower_joint_limits = self._ensure_tensor(lower_limits)
        self.upper_joint_limits = self._ensure_tensor(upper_limits)
        self.lower_joint_limits_np = self.lower_joint_limits.detach().cpu().numpy()[0]
        self.upper_joint_limits_np = self.upper_joint_limits.detach().cpu().numpy()[0]

    def _load_meshes(self, load_visual: bool, load_collision: bool):
        """Load visual and collision meshes for the robot."""
        from utils.mesh_and_urdf_utils import load_link_geometries
        
        self.link_visuals_dict = {}
        self.link_collisions_dict = {}
        
        link_names = self.pk_chain.get_link_names()
        
        if load_visual:
            self.link_visuals_dict = load_link_geometries(
                str(self.urdf_path), link_names, collision=False
            )
        
        if load_collision:
            self.link_collisions_dict = load_link_geometries(
                str(self.urdf_path), link_names, collision=True
            )

    def set_reference_joint_values(self):
        """Set reference joint values (midpoint between limits by default)."""
        self.reference_joint_values = (self.lower_joint_limits + self.upper_joint_limits) / 2
        self.reference_joint_values_np = self.reference_joint_values.detach().cpu().numpy()[0]

    def get_state_trimesh(
        self,
        joint_pos: Union[list, np.ndarray, torch.Tensor],
        X_w_b: torch.Tensor = torch.eye(4),
        visual: bool = True,
        collision: bool = False,
    ) -> Dict[str, trimesh.Scene]:
        """Get trimesh representation of the robot in specified configuration.
        
        Args:
            joint_pos: Joint positions (list, np.array or torch.Tensor)
            X_w_b: 4x4 transformation matrix for base pose in world frame
            visual: Whether to include visual mesh
            collision: Whether to include collision mesh
            
        Returns:
            Dictionary containing requested scenes ('visual' and/or 'collision')
        """
        return_dict = {}
        joint_pos_tensor = self._ensure_tensor(joint_pos)
        
        # Compute forward kinematics
        current_status = self.pk_chain.forward_kinematics(th=joint_pos_tensor)
        
        if visual and hasattr(self, 'link_visuals_dict'):
            return_dict["visual"] = self._build_scene(
                current_status, X_w_b, self.link_visuals_dict
            )
            
        if collision and hasattr(self, 'link_collisions_dict'):
            return_dict["collision"] = self._build_scene(
                current_status, X_w_b, self.link_collisions_dict
            )
            
        return return_dict

    def _build_scene(
        self,
        fk_result: Dict[str, pk.Transform3d],
        base_transform: torch.Tensor,
        mesh_dict: Dict[str, trimesh.Trimesh]
    ) -> trimesh.Scene:
        """Helper method to build a trimesh scene from meshes and transforms."""
        scene = trimesh.Scene()
        
        for link_name, mesh in mesh_dict.items():
            transform = (base_transform @ 
                        fk_result[link_name].get_matrix().detach().cpu().numpy().reshape(4, 4))
            scene.add_geometry(mesh.copy().apply_transform(transform))
            
        return scene

    def _ensure_tensor(
        self, 
        th: Union[list, np.ndarray, torch.Tensor], 
        ensure_batch_dim: bool = True
    ) -> torch.Tensor:
        """Convert input to properly shaped tensor.
        
        Args:
            th: Input (list, np.array or torch.Tensor)
            ensure_batch_dim: Whether to add batch dimension if missing
            
        Returns:
            Tensor on correct device with proper shape
        """
        if isinstance(th, (np.ndarray, list)):
            th = torch.tensor(th, device=self.device, dtype=self.dtype)
            
        if ensure_batch_dim and len(th.shape) < 2:
            th = th.unsqueeze(0)
            
        return th


class XArm7PK(RobotArmPK):
    """Specialized class for xArm7"""
    
    def __init__(
        self,
        urdf_path: Optional[Path] = None,
        load_visual_mesh: bool = True,
        load_col_mesh: bool = True,
        dtype=torch.float,
        device="cuda",
    ):
        """Initialize xArm7 model.
        
        Args:
            urdf_path: Optional custom URDF path (uses default if None)
            Other args same as parent class
        """
        if urdf_path is None:
            raise ValueError("URDF path must be provided for xArm7.")
        super().__init__(
            urdf_path=urdf_path,
            load_visual_mesh=load_visual_mesh,
            load_col_mesh=load_col_mesh,
            dtype=dtype,
            device=device,
        )

    def set_reference_joint_values(self):
        """Set zero position as reference for xArm7."""
        self.reference_joint_values = torch.zeros_like(self.lower_joint_limits)
        self.reference_joint_values_np = self.reference_joint_values.detach().cpu().numpy()[0]


"""Example of creating a new robot type:"""
class MyCustomArm(RobotArmPK):
    def __init__(self, urdf_path=None, **kwargs):
        if urdf_path is None:
            raise ValueError("URDF path must be provided for MyCustomArm.")
        super().__init__(urdf_path, **kwargs)
    
    def set_reference_joint_values(self):
        # Custom implementation if needed
        pass
    
    """
    Normally you would only need to override 
    set_reference_joint_values 
    based on the default configuration of your custom arm.
    """

