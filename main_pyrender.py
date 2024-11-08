import trimesh
import pyrender
import numpy as np
import cv2
import argparse
import os


def convert_scene_to_single_mesh(scene):
    """ Convert a trimesh Scene object into a single mesh. """
    meshes = []

    # If the scene is not empty, combine all geometries into one mesh
    for m in scene.geometry.values():
        meshes.append(m)

    # Combine all meshes into a single mesh
    if meshes:
        combined_mesh = trimesh.util.concatenate(meshes)
        return combined_mesh
    else:
        raise ValueError("The scene does not contain any geometry to merge.")


def load_mesh_and_create_video(mesh_path, output_video='rotation_video.mp4', frames=60, resolution=(640, 480), rotation_speed=0.5):
    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # If the mesh is a scene, convert it to a single mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = convert_scene_to_single_mesh(mesh)

    # Scale and center the mesh
    mesh_extents = mesh.extents
    mesh_center = mesh.centroid

    # Scale factor to fit the mesh within a unit sphere and Apply scaling and centering transformations
    scale_factor = 1.0 / np.max(mesh_extents)
    mesh.apply_translation(-mesh_center)  # Center the mesh
    mesh.apply_scale(scale_factor)  # Scale the mesh

    # # Rotate the mesh to swap Z and Y axes
    # rotation_matrix = trimesh.transformations.rotation_matrix(
    #     angle=np.pi / 2,  # 90 degrees
    #     direction=[-1, 0, 0],  # Rotate around the X-axis
    #     point=[0, 0, 0]
    # )
    # mesh.apply_transform(rotation_matrix)

    # Set up the pyrender scene
    scene = pyrender.Scene()

    # Create a diffusive material
    material = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        roughnessFactor=1.0,
        baseColorFactor=[0.9, 0.9, 0.9, 1.0]
    )

    mesh_pyrender = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(mesh_pyrender)

    # Create a perspective camera and add it to the scene
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    cam_pose = np.eye(4)
    cam_node = scene.add(camera, pose=cam_pose)

    # Create a light and add it to the scene
    light1 = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    light_node = scene.add(light1, pose=np.eye(4))

    # Set up video writer with OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, 30, resolution)

    # Set up a viewer and render
    r = pyrender.OffscreenRenderer(*resolution)

    # Camera rotation around the object (fixed at the origin)
    camera_distance = 1.5  # Camera's fixed distance from the object
    for i in range(frames):
        angle = i * (2 * np.pi / frames) * rotation_speed  # Adjusted angle per frame

        # Create the camera's rotation matrix around the Y-axis (fixed distance from the object)
        cam_pose = np.array([
            [np.cos(angle), 0, np.sin(angle), camera_distance * np.sin(angle)],  # X position
            [0, 1, 0, 0],  # Y position remains fixed
            [-np.sin(angle), 0, np.cos(angle), camera_distance * np.cos(angle)],  # Z position
            [0, 0, 0, 1]  # Homogeneous coordinate
        ])

        # Ensure the camera is always looking at the center of the object (the origin)
        scene.set_pose(light_node, pose=cam_pose)
        scene.set_pose(cam_node, pose=cam_pose)

        # Render the scene
        color, _ = r.render(scene)
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(color)

    # Release the video writer and renderer
    video_writer.release()
    r.delete()

    print(f'Video saved as {output_video}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a video of a rotating mesh')
    parser.add_argument('--mesh_path', type=str, default='./meshes/lucy.glb', help='Path to the mesh file')
    parser.add_argument('--output_video', type=str, default='rotation_video.mp4', help='Path to the output video')
    parser.add_argument('--frames', type=int, default=240, help='Number of frames in the video')
    parser.add_argument('--resolution', type=int, nargs=2, default=(1024, 1024), help='Resolution of the video')
    parser.add_argument('--rotation_speed', type=float, default=1.0, help='Speed of rotation')
    args = parser.parse_args()

    output_video = os.path.join('./outputs', args.mesh_path.split('/')[-1].split('.')[0] + '.mp4')
    os.makedirs('./outputs', exist_ok=True)
    load_mesh_and_create_video(args.mesh_path, output_video, args.frames, args.resolution, args.rotation_speed)