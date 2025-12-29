import os
import viser
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(
                    trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()
                )
            )
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh


def parse_origin(element):
    """Parse the origin element for translation and rotation."""
    origin = element.find("origin")
    xyz = np.zeros(3)
    rotation = np.eye(3)
    if origin is not None:
        xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
        rpy = np.fromstring(
            origin.attrib.get("rpy", "0 0 0"), sep=" "
        )  # Roll, pitch, yaw
        rotation = R.from_euler("xyz", rpy).as_matrix()
    return xyz, rotation


def apply_transform(mesh, translation, rotation):
    """Apply translation and rotation to a mesh."""
    # mesh.apply_translation(-mesh.centroid)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    mesh.apply_transform(transform)
    return mesh


def create_primitive_mesh(geometry, translation, rotation):
    """Create a trimesh object from primitive geometry definitions with transformations."""
    if geometry.tag.endswith("box"):
        size = np.fromstring(geometry.attrib["size"], sep=" ")
        mesh = trimesh.creation.box(extents=size)
    elif geometry.tag.endswith("sphere"):
        radius = float(geometry.attrib["radius"])
        mesh = trimesh.creation.icosphere(radius=radius)
    elif geometry.tag.endswith("cylinder"):
        radius = float(geometry.attrib["radius"])
        length = float(geometry.attrib["length"])
        mesh = trimesh.creation.cylinder(radius=radius, height=length)
    else:
        raise ValueError(f"Unsupported geometry type: {geometry.tag}")
    return apply_transform(mesh, translation, rotation)


def extract_colors_from_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    global_materials = {}
    for material in root.findall("material"):
        name = material.attrib["name"]
        color_elem = material.find("color")
        if color_elem is not None and "rgba" in color_elem.attrib:
            rgba = [float(c) for c in color_elem.attrib["rgba"].split()]
            global_materials[name] = rgba

    link_colors = {}

    for link in root.iter("link"):
        link_name = link.attrib["name"]
        visual = link.find("./visual")
        if visual is not None:
            material = visual.find("./material")
            if material is not None:
                color = material.find("color")
                if color is not None and "rgba" in color.attrib:
                    rgba = [float(c) for c in color.attrib["rgba"].split()]
                    link_colors[link_name] = rgba
                elif "name" in material.attrib:
                    material_name = material.attrib["name"]
                    if material_name in global_materials:
                        link_colors[link_name] = global_materials[material_name]

    return link_colors


def load_link_geometries(urdf_path, link_names, collision=False):
    """Load geometries (trimesh objects) for specified links from a URDF file, considering origins."""
    urdf_dir = os.path.dirname(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_geometries = {}
    link_colors_from_urdf = extract_colors_from_urdf(urdf_path)

    for link in root.findall("link"):
        link_name = link.attrib["name"]
        link_color = link_colors_from_urdf.get(link_name, None)
        if link_name in link_names:
            geom_index = "visual"
            if collision:
                geom_index = "collision"
            link_mesh = []
            for visual in link.findall(".//" + geom_index):
                geometry = visual.find("geometry")
                xyz, rotation = parse_origin(visual)
                try:
                    if geometry[0].tag.endswith("mesh"):
                        mesh_filename = geometry[0].attrib["filename"]
                        full_mesh_path = os.path.join(urdf_dir, mesh_filename)
                        mesh = as_mesh(trimesh.load(full_mesh_path))
                        scale = np.fromstring(geometry[0].attrib.get("scale", "1 1 1"), sep=" ")
                        mesh.apply_scale(scale)
                        link_mesh.append(apply_transform(mesh, xyz, rotation))
                    else:  # Handle primitive shapes
                        mesh = create_primitive_mesh(geometry[0], xyz, rotation)
                        link_mesh.append(mesh)

                except Exception as e:
                    print(f"Failed to load geometry for {link_name}: {e}")
            if len(link_mesh) == 0:
                continue
            elif len(link_mesh) > 1:
                link_trimesh = as_mesh(trimesh.Scene(link_mesh))
            elif len(link_mesh) == 1:
                link_trimesh = link_mesh[0]
                
            if link_color is not None:
                link_trimesh.visual.face_colors = np.array(link_color)
            link_geometries[link_name] = link_trimesh

    return link_geometries


def vis_robot_frames(
    server,
    current_status,
    axes_length=0.05,
    axes_radius=0.0025,
):
    """
    Visualize the robot frames.
    """

    for link_name, link_transform in current_status.items():
        link_transform_np = (
            link_transform.get_matrix().detach().cpu().numpy().reshape(4, 4)
        )
        link_transform_quat_wxyz = trimesh.transformations.quaternion_from_matrix(
            link_transform_np
        )
        link_transform_xyz = link_transform_np[:3, 3]

        server.scene.add_frame(
            "frame_" + link_name,
            wxyz=link_transform_quat_wxyz,
            position=link_transform_xyz,
            axes_length=axes_length,
            axes_radius=axes_radius,
        )


def update_viser_mp_result(sv:viser.ViserServer, arm_pk, current_joint_values, target_joint_values, waypt_joint_values_np, n_vis_waypt=5):
    # Load the current and target arm meshes
    current_arm_mesh = arm_pk.get_state_trimesh(current_joint_values, visual=True, collision=False)["visual"]
    target_arm_mesh = as_mesh(arm_pk.get_state_trimesh(target_joint_values, visual=True, collision=False)["visual"])

    # Add the current mesh to the scene
    sv.scene.add_mesh_trimesh("current_arm_mesh", current_arm_mesh)

    # Get the 'rainbow' colormap
    colormap = plt.get_cmap('rainbow')

    # Set color for the target mesh using the 'rainbow' colormap (choose a specific color, e.g., 0.8)
    target_color = colormap(0.9)[:3]  # Extract RGB values from the colormap

    # Add the target mesh to the scene with the specified color
    sv.scene.add_mesh_simple("target_arm_mesh", target_arm_mesh.vertices, target_arm_mesh.faces, color=target_color, opacity=0.9)

    # Return if there are no waypoints
    if len(waypt_joint_values_np) == 0:
        return

    # Determine waypoints to visualize
    n_total_waypt = len(waypt_joint_values_np)  # Total number of waypoints
    vis_waypt_idx = np.linspace(0, n_total_waypt - 1, num=n_vis_waypt + 2, dtype=int)[1:-1]
    vis_waypt_joint_values_np = waypt_joint_values_np[vis_waypt_idx]

    # Generate colors for each waypoint from the 'rainbow' colormap
    colors = colormap(np.linspace(0.1, 0.9, len(vis_waypt_joint_values_np)))  # Generate colors for each waypoint

    # Add each waypoint mesh to the scene with a color from the colormap
    for waypt_idx, vis_waypt_joint_values in enumerate(vis_waypt_joint_values_np):
        vis_waypt_mesh = as_mesh(arm_pk.get_state_trimesh(vis_waypt_joint_values, visual=True, collision=False)["visual"])
        waypoint_color = colors[waypt_idx][:3]  # Extract RGB values for this waypoint

        sv.scene.add_mesh_simple(f"vis_waypt_mesh_{waypt_idx}", vis_waypt_mesh.vertices, vis_waypt_mesh.faces, color=waypoint_color, opacity=0.5)