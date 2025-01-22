import numpy as np
import cv2


class PCDSphericalConverter:
    def __init__(self, width: int = 1024, height: int = 512):
        """
        Initialize the spherical image converter.

        Args:
            width: Output equirectangular image width
            height: Output equirectangular image height
        """
        self.width = width
        self.height = height
        self._initialize_spherical_grid()

    def _initialize_spherical_grid(self):
        """Create spherical coordinate grid for the output image."""
        x = np.linspace(0, 2 * np.pi, self.width)
        y = np.linspace(0, np.pi, self.height)

        theta, phi = np.meshgrid(x, y)

        # Convert spherical coordinates to Cartesian
        self.x = np.sin(phi) * np.cos(theta)
        self.y = np.sin(phi) * np.sin(theta)
        self.z = np.cos(phi)

    def project_points(self, points_3d: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """
        Project 3D points onto spherical surface.

        Args:
            points_3d: Nx3 array of 3D points
            colors: Nx3 array of RGB colors for each point

        Returns:
            Spherical image as numpy array (height x width x 3)
        """
        # Initialize output image
        spherical_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Convert 3D points to spherical coordinates
        r = np.sqrt(np.sum(points_3d ** 2, axis=1))
        theta = np.arctan2(points_3d[:, 1], points_3d[:, 0])
        phi = np.arccos(points_3d[:, 2] / r)

        # Map to pixel coordinates
        x_px = ((theta + np.pi) / (2 * np.pi) * self.width).astype(int)
        y_px = (phi / np.pi * self.height).astype(int)

        # Filter valid pixels
        valid_mask = (x_px >= 0) & (x_px < self.width) & (y_px >= 0) & (y_px < self.height)

        # Assign colors to pixels
        spherical_img[y_px[valid_mask], x_px[valid_mask]] = colors[valid_mask]

        return spherical_img

    def process_point_cloud(self,
                            points: np.ndarray,
                            colors: np.ndarray,
                            camera_position: np.ndarray = None) -> np.ndarray:
        """
        Process a point cloud to create a spherical image.

        Args:
            points: Nx3 array of points in 3D space
            colors: Nx3 array of RGB colors
            camera_position: Optional 3D position of the camera

        Returns:
            Equirectangular projection as numpy array
        """
        if camera_position is not None:
            # Transform points relative to camera position
            points = points - camera_position

        # Project points to sphere
        spherical_image = self.project_points(points, colors)

        # Fill holes using interpolation
        mask = np.all(spherical_image == 0, axis=2)
        for channel in range(3):
            channel_data = spherical_image[:, :, channel]
            channel_data[mask] = cv2.inpaint(
                channel_data,
                mask.astype(np.uint8),
                3,
                cv2.INPAINT_TELEA
            )[mask]

        return spherical_image
