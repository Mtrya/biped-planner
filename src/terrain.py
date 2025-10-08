"""
Terrain module for bipedal robot footstep planning.

Handles 1000x3000 height map loading and bilinear interpolation.
"""

import numpy as np
import os
from typing import Tuple


class Terrain:
    """
    Simple terrain handler with bilinear interpolation.

    Attributes:
        height_map (np.ndarray): 1000x3000 height map array
        height (int): Map height (1000)
        width (int): Map width (3000)
    """

    def __init__(self, terrain_path: str = "data/terrain_map.npy"):
        """Initialize terrain, load existing map or generate new one."""
        self.height = 1000
        self.width = 3000

        if os.path.exists(terrain_path):
            self.height_map = np.load(terrain_path)
            if self.height_map.shape != (self.height, self.width):
                raise ValueError(f"Terrain shape {self.height_map.shape} != expected ({self.height}, {self.width})")
        else:
            self.height_map = self._generate_terrain()
            self._save_terrain(terrain_path)

    def get_height(self, x: float, y: float) -> float:
        """
        Get terrain height at continuous (x, y) coordinates using bilinear interpolation.

        Args:
            x: x coordinate in [0, width-1]
            y: y coordinate in [0, height-1]

        Returns:
            Height value at (x, y)
        """
        # Clamp to valid range
        x = np.clip(x, 0, self.width - 1)
        y = np.clip(y, 0, self.height - 1)

        # Get integer coordinates
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.width - 1), min(y0 + 1, self.height - 1)

        # Get fractional parts
        fx, fy = x - x0, y - y0

        # Bilinear interpolation
        h00 = self.height_map[y0, x0]
        h10 = self.height_map[y0, x1]
        h01 = self.height_map[y1, x0]
        h11 = self.height_map[y1, x1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fy) + h1 * fy

    def get_gradient(self, x: float, y: float) -> Tuple[float, float]:
        """
        Get terrain gradient at (x, y) using finite differences.

        Args:
            x: x coordinate in [0, width-1]
            y: y coordinate in [0, height-1]

        Returns:
            (dh/dx, dh/dy) gradient tuple
        """
        # Finite difference approximation
        dx = 1.0
        dy = 1.0

        # Handle boundaries
        x_left = max(x - dx, 0)
        x_right = min(x + dx, self.width - 1)
        y_bottom = max(y - dy, 0)
        y_top = min(y + dy, self.height - 1)

        dh_dx = (self.get_height(x_right, y) - self.get_height(x_left, y)) / (2 * dx)
        dh_dy = (self.get_height(x, y_top) - self.get_height(x, y_bottom)) / (2 * dy)

        return dh_dx, dh_dy

    def _generate_terrain(self) -> np.ndarray:
        """
        Generate terrain using MATLAB logic translated to Python.

        Creates horizontal bands with Gaussian noise.
        """
        # Initialize with ones
        terrain = np.ones((self.height, self.width), dtype=np.float32)

        # Base heights following MATLAB logic: h0=30, then arithmetic progression
        h0 = 30.0
        heights = [h0, h0 + h0, h0 + 2*h0, h0 + 3*h0, h0 + 2*h0, h0 + h0, h0]
        # heights = [30, 60, 90, 120, 90, 60, 30]

        # Define column ranges (0-indexed in Python)
        ranges = [
            (0, 999),      # Columns 1-1000 in MATLAB
            (999, 1059),   # Columns 1000-1060 in MATLAB
            (1059, 1119),  # Columns 1060-1120 in MATLAB
            (1119, 1299),  # Columns 1120-1300 in MATLAB
            (1299, 1499),  # Columns 1300-1500 in MATLAB
            (1499, 1999),  # Columns 1500-2000 in MATLAB
            (1999, 3000)   # Columns 2000-3000 in MATLAB
        ]

        # Fill terrain with base heights
        for (start, end), height in zip(ranges, heights):
            terrain[:, start:end] = height

        # Add Gaussian noise (randn in MATLAB)
        noise = np.random.randn(self.height, self.width)
        terrain = terrain + noise

        # Convert to uint8 and clip to [0, 255]
        terrain = np.clip(terrain, 0, 255).astype(np.uint8)

        return terrain

    def _save_terrain(self, path: str) -> None:
        """Save terrain map to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, self.height_map)


# Global terrain instance for easy access
_terrain_instance = None

def get_terrain(terrain_path: str = "data/terrain_map.npy") -> Terrain:
    """Get or create terrain instance (singleton pattern)."""
    global _terrain_instance
    if _terrain_instance is None:
        _terrain_instance = Terrain(terrain_path)
    return _terrain_instance

def get_height(x: float, y: float) -> float:
    """Convenience function: get height at (x, y)."""
    return get_terrain().get_height(x, y)

def get_gradient(x: float, y: float) -> Tuple[float, float]:
    """Convenience function: get gradient at (x, y)."""
    return get_terrain().get_gradient(x, y)


if __name__ == "__main__":
    # Test terrain generation and interpolation
    terrain = Terrain()

    print(f"Terrain shape: {terrain.height_map.shape}")
    print(f"Height range: [{terrain.height_map.min()}, {terrain.height_map.max()}]")

    # Test interpolation at some points
    test_points = [(100.5, 200.3), (500.0, 500.0), (999.9, 2999.9)]
    for x, y in test_points:
        h = terrain.get_height(x, y)
        grad_x, grad_y = terrain.get_gradient(x, y)
        print(f"Height at ({x:.1f}, {y:.1f}): {h:.2f}, gradient: ({grad_x:.3f}, {grad_y:.3f})")