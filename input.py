import numpy as np
import open3d as o3d
from typing import Tuple


def example_point_cloud() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an example point cloud of a simple scene with:
    - Ground plane
    - Two walls
    - Some floating spheres

    Returns:
        Tuple of (points, colors) where:
        - points is an Nx3 array of 3D coordinates
        - colors is an Nx3 array of RGB values
    """
    points_list = []
    colors_list = []

    # Create ground plane
    x = np.linspace(-5, 5, 100)
    z = np.linspace(-5, 5, 100)
    X, Z = np.meshgrid(x, z)
    Y = np.zeros_like(X)

    ground_points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    # Checkerboard pattern for ground
    checker = (np.floor(X / 2) + np.floor(Z / 2)) % 2
    ground_colors = np.where(checker.flatten()[:, None] == 0,
                             np.array([200, 200, 200]),
                             np.array([100, 100, 100]))

    points_list.append(ground_points)
    colors_list.append(ground_colors)

    # Create walls
    wall_points = []
    wall_colors = []

    # Wall 1 (red)
    x = np.linspace(-5, 5, 50)
    y = np.linspace(0, 4, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, -5)
    wall_points.append(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1))
    wall_colors.append(np.tile([180, 100, 100], (len(X.flatten()), 1)))

    # Wall 2 (blue)
    z = np.linspace(-5, 5, 50)
    y = np.linspace(0, 4, 50)
    Z, Y = np.meshgrid(z, y)
    X = np.full_like(Z, 5)
    wall_points.append(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1))
    wall_colors.append(np.tile([100, 100, 180], (len(Z.flatten()), 1)))

    points_list.extend(wall_points)
    colors_list.extend(wall_colors)

    # Add some floating spheres
    sphere_centers = [
        (-2, 2, -2, [255, 50, 50]),  # Red sphere
        (2, 1, 2, [50, 255, 50]),  # Green sphere
        (0, 3, 0, [50, 50, 255]),  # Blue sphere
    ]

    for x, y, z, color in sphere_centers:
        # Create sphere points
        phi = np.linspace(0, 2 * np.pi, 20)
        theta = np.linspace(0, np.pi, 20)
        PHI, THETA = np.meshgrid(phi, theta)

        r = 0.5  # sphere radius
        x_s = x + r * np.sin(THETA) * np.cos(PHI)
        y_s = y + r * np.sin(THETA) * np.sin(PHI)
        z_s = z + r * np.cos(THETA)

        sphere_points = np.stack([x_s.flatten(), y_s.flatten(), z_s.flatten()], axis=1)
        sphere_colors = np.tile(color, (len(sphere_points), 1))

        points_list.append(sphere_points)
        colors_list.append(sphere_colors)

    # Combine all points and colors
    points = np.vstack(points_list)
    colors = np.vstack(colors_list)

    return points, colors


def load_point_cloud(view=False) -> Tuple[np.ndarray, np.ndarray]:
    # pcd_data = o3d.data.PLYPointCloud()
    # pcd = o3d.io.read_point_cloud(pcd_data.path)
    pcd = o3d.io.read_point_cloud('data/room.ply')

    points = np.asarray(pcd.points)
    has_color = pcd.has_colors()
    if has_color:
        colors = np.asarray(pcd.colors)
        colors = (colors / np.max(colors) * 255).astype(np.int64)

    if view:
        o3d.visualization.draw_geometries([pcd])
    return points, colors
