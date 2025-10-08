import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    from .terrain import get_terrain
except ImportError as e:
    from terrain import get_terrain


def plot_terrain_3d(grid_size=8):
    """Load terrain and plot it in 3D."""
    terrain = get_terrain()

    # Create grid
    x = np.arange(0, terrain.height, grid_size)
    y = np.arange(0, terrain.width, grid_size)
    X, Y = np.meshgrid(x, y)

    # Sample heights
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = terrain.get_height(X[i, j], Y[i, j])

    # Plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='terrain')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Height')
    ax.set_title(f'Terrain (grid: {grid_size}x{grid_size})')

    plt.show()


if __name__ == "__main__":
    plot_terrain_3d()