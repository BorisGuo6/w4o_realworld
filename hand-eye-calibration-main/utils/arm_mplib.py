from __future__ import annotations

import sapien.core as sapien
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

@dataclass
class MPlibConfig:
    urdf_path: Path = None

    vis: bool = True
    
    timestep: float = 1 / 240.0
    static_friction: float = 1.0
    dynamic_friction: float = 1.0 
    restitution: float = 0.0
    ambient_light: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    shadow: bool = True
    direction_lights: List[List[List[float]]] = field(default_factory=lambda: [[[0, 1, -1], [0.5, 0.5, 0.5]]])
    point_lights: List[List[List[float]]] = field(default_factory=lambda: [
        [[1, 2, 2], [1, 1, 1]],
        [[1, -2, 2], [1, 1, 1]],
        [[-1, 0, 1], [1, 1, 1]]
    ])

    joint_stiffness: float = 1000
    joint_damping: float = 200

    def get(self, key, default):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default
    
class RobotMPlib:
    def __init__(self, cfg: MPlibConfig):
        self.cfg = cfg
        self.scene = sapien.Scene()
        if cfg.vis:
            self.set_viewer()

        # set simulation timestep
        self.scene.set_timestep(self.cfg.timestep)
        # set default physical material
        self.scene.default_physical_material = self.scene.create_physical_material(
            self.cfg.static_friction,
            self.cfg.dynamic_friction,
            self.cfg.restitution,
        )
        
        loader: sapien.URDFLoader = self.scene.create_urdf_loader()
        loader.fix_root_link = True
        self.sp_robot: sapien.Articulation = loader.load(str(cfg.urdf_path))
        self.sp_robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        self.active_joints = self.sp_robot.get_active_joints()
        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness=self.cfg.get("joint_stiffness", 1000),
                damping=self.cfg.get("joint_damping", 200),
            )

        # give some white ambient light of moderate intensity
        self.scene.set_ambient_light(self.cfg.ambient_light)
        # default enable shadow unless specified otherwise
        shadow = self.cfg.shadow
        # default spotlight angle and intensity
        direction_lights = self.cfg.direction_lights
        for direction_light in direction_lights:
            self.scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        # default point lights position and intensity
        point_lights = self.cfg.point_lights
        for point_light in point_lights:
            self.scene.add_point_light(point_light[0], point_light[1], shadow=shadow)