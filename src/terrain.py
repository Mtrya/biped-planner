"""
Terrain module for bipedal robot footstep planning.

Handles 3000x1000 height map loading and bilinear interpolation.
"""

import numpy as np
import os
from typing import Tuple


class Terrain:
    """
    Simple terrain handler with bilinear interpolation.

    Attributes:
        height_map (np.ndarray): 3000x1000 height map array (height x width)
        height (int): Map height (3000, X dimension)
        width (int): Map width (1000, Y dimension)
    """

    def __init__(self, terrain_path: str = "data/terrain_map.npy"):
        """Initialize terrain, load existing map or generate new one."""
        self.height = 3000  # x ∈ [0,3000]
        self.width = 1000   # y ∈ [0,1000]

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
            x: x coordinate in [0, height-1]
            y: y coordinate in [0, width-1]

        Returns:
            Height value at (x, y)
        """
        # Clamp to valid range
        x = np.clip(x, 0, self.height - 1)
        y = np.clip(y, 0, self.width - 1)

        # Get integer coordinates
        x0, y0 = int(x), int(y)
        x1, y1 = min(x0 + 1, self.height - 1), min(y0 + 1, self.width - 1)

        # Get fractional parts
        fx, fy = x - x0, y - y0

        # Bilinear interpolation
        h00 = self.height_map[x0, y0]
        h10 = self.height_map[x1, y0]
        h01 = self.height_map[x0, y1]
        h11 = self.height_map[x1, y1]

        h0 = h00 * (1 - fx) + h10 * fx
        h1 = h01 * (1 - fx) + h11 * fx

        return h0 * (1 - fy) + h1 * fy

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
            (0, 999),
            (999, 1059),
            (1059, 1119),
            (1119, 1299),
            (1299, 1499),
            (1499, 1999),
            (1999, 3000)
        ]

        # Fill terrain with base heights
        for (start, end), height in zip(ranges, heights):
            terrain[start:end, :] = height

        # Add Gaussian noise (randn in MATLAB)
        noise_scale = 0.1
        noise = noise_scale * np.random.randn(self.height, self.width)
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


if __name__ == "__main__":
    # Test terrain generation and interpolation
    terrain = Terrain()

    print(f"Terrain shape: {terrain.height_map.shape}")
    print(f"Height range: [{terrain.height_map.min()}, {terrain.height_map.max()}]")

    # Test interpolation at some points
    test_points = [(100.5, 200.3), (500.0, 500.0), (999.9, 2999.9)]
    for x, y in test_points:
        h = terrain.get_height(x, y)
        print(f"Height at ({x:.1f}, {y:.1f}): {h:.2f}")