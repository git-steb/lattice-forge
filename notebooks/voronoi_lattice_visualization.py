#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

#%%
# Function to generate lattice points
def generate_lattice_points(x_range, y_range, step=1):
    """Generate lattice points within the specified x and y ranges."""
    x_coords = np.arange(x_range[0], x_range[1] + step, step)
    y_coords = np.arange(y_range[0], y_range[1] + step, step)
    points = np.array(np.meshgrid(x_coords, y_coords)).T.reshape(-1, 2)
    return points

#%%
# Function to plot Voronoi diagram for lattice points
def plot_lattice_voronoi(x_range=(-5, 5), y_range=(-5, 5), step=1):
    """Generate and plot the Voronoi diagram for a 2D lattice grid."""
    points = generate_lattice_points(x_range, y_range, step)
    vor = Voronoi(points)

    # Plot the Voronoi diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6, point_size=5)
    ax.plot(points[:, 0], points[:, 1], 'ro', markersize=4, label='Lattice Points')

    # Customize the plot
    ax.set_xlim(x_range[0] - 1, x_range[1] + 1)
    ax.set_ylim(y_range[0] - 1, y_range[1] + 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Voronoi Diagram of Lattice Points')
    ax.legend()
    ax.grid(True)

    plt.show()

#%%
# Generate and display the Voronoi diagram
plot_lattice_voronoi(x_range=(-5, 5), y_range=(-5, 5), step=1)
