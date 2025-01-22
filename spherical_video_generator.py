# Generate a perspective view from the spherical image
import numpy as np
import cv2
from typing import Tuple


class SphericalVideoGenerator:
    def __init__(self, spherical_image: np.ndarray):
        """
        Initialize the video generator with a spherical (equirectangular) image.

        Args:
            spherical_image: HxWx3 numpy array containing the equirectangular image
        """
        self.spherical_image = spherical_image
        self.height, self.width = spherical_image.shape[:2]

    def get_perspective_view(self,
                             yaw: float,
                             pitch: float,
                             fov: float,
                             output_size: Tuple[int, int]) -> np.ndarray:
        """
        Generate a perspective view from the spherical image.

        Args:
            yaw: Horizontal rotation angle in radians
            pitch: Vertical rotation angle in radians
            fov: Field of view in degrees
            output_size: (width, height) of the output image

        Returns:
            Perspective view as numpy array
        """
        out_h, out_w = output_size

        # Create meshgrid for output image
        x = np.linspace(0, out_w - 1, out_w)
        y = np.linspace(0, out_h - 1, out_h)
        X, Y = np.meshgrid(x, y)

        # Convert to normalized device coordinates
        X = (X - out_w / 2) / (out_w / 2)
        Y = (Y - out_h / 2) / (out_h / 2)

        # Calculate ray directions
        fov_rad = np.radians(fov)
        Z = np.ones_like(X) / np.tan(fov_rad / 2)

        # Create rotation matrices
        rot_pitch = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])

        rot_yaw = np.array([
            [np.cos(yaw), 0, np.sin(yaw)],
            [0, 1, 0],
            [-np.sin(yaw), 0, np.cos(yaw)]
        ])

        # Combine rotations
        rotation = rot_yaw @ rot_pitch

        # Stack coordinates and reshape
        xyz = np.stack([X, Y, Z], axis=-1)
        points = xyz.reshape(-1, 3)

        # Apply rotation
        rotated = (rotation @ points.T).T

        # Convert to spherical coordinates
        theta = np.arctan2(rotated[:, 0], rotated[:, 2])
        phi = np.arctan2(np.sqrt(rotated[:, 0] ** 2 + rotated[:, 2] ** 2), rotated[:, 1])

        # Map to spherical image coordinates
        u = ((theta + np.pi) / (2 * np.pi)) * self.width
        v = (phi / np.pi) * self.height

        # Interpolate colors
        u = np.clip(u, 0, self.width - 1)
        v = np.clip(v, 0, self.height - 1)

        u_0 = np.floor(u).astype(int)
        u_1 = np.ceil(u).astype(int)
        v_0 = np.floor(v).astype(int)
        v_1 = np.ceil(v).astype(int)

        # Bilinear interpolation weights
        w_u = u - u_0
        w_v = v - v_0

        # Sample colors
        c_00 = self.spherical_image[v_0, u_0]
        c_01 = self.spherical_image[v_0, u_1]
        c_10 = self.spherical_image[v_1, u_0]
        c_11 = self.spherical_image[v_1, u_1]

        # Interpolate
        c_0 = c_00 * (1 - w_u[:, None]) + c_01 * w_u[:, None]
        c_1 = c_10 * (1 - w_u[:, None]) + c_11 * w_u[:, None]
        colors = c_0 * (1 - w_v[:, None]) + c_1 * w_v[:, None]
        frame = colors.reshape(out_h, out_w, 3)
        frame = cv2.flip(frame, 0)
        return frame.astype(np.uint8)

    def generate_rotation_video(self,
                                output_path: str,
                                duration_seconds: float = 10.0,
                                fps: int = 30,
                                view_fov: float = 90.0,
                                output_size: Tuple[int, int] = (720, 1280)):
        """
        Generate a video rotating around the scene.

        Args:
            output_path: Path to save the output video
            duration_seconds: Duration of the video in seconds
            fps: Frames per second
            view_fov: Field of view in degrees
            output_size: (height, width) of the output video
        """
        n_frames = int(duration_seconds * fps)

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (output_size[1], output_size[0]))

        for i in range(n_frames):
            # Calculate rotation angles
            progress = i / n_frames
            yaw = 2 * np.pi * progress  # Full 360° rotation
            pitch = np.sin(progress * 2 * np.pi) * 0.3  # Gentle up/down motion

            # Generate frame
            frame = self.get_perspective_view(
                yaw=yaw,
                pitch=pitch,
                fov=view_fov,
                output_size=output_size
            )

            # Write frame
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
