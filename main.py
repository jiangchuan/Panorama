import numpy as np
import cv2
from input import example_point_cloud, load_point_cloud
from pcd_spherical_converter import PCDSphericalConverter
from spherical_video_generator import SphericalVideoGenerator

# Load point cloud
# points, colors = example_point_cloud()
points, colors = load_point_cloud()

# Initialize converter
converter = PCDSphericalConverter(width=2048, height=1024)

# Set camera position slightly elevated and back from the scene center
camera_position = np.mean(points, axis=0)

# Convert to spherical image
spherical_image = converter.process_point_cloud(
    points=points,
    colors=colors,
    camera_position=camera_position
)

# Save result
image_path = 'data/spherical_room.jpg'
cv2.imwrite(image_path, cv2.cvtColor(spherical_image, cv2.COLOR_RGB2BGR))

# Load the spherical image
spherical_image = cv2.imread(image_path)
spherical_image = cv2.cvtColor(spherical_image, cv2.COLOR_BGR2RGB)

# Create video generator
generator = SphericalVideoGenerator(spherical_image)

# Generate rotating video
fov = 90
generator.generate_rotation_video(
    output_path=f'data/rotating_room_{fov}.mp4',
    duration_seconds=10.0,
    fps=30,
    view_fov=fov,
    output_size=(720, 1280)  # 720p portrait orientation
)
