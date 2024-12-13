#%% [markdown]
# # Voronoi Lattice Visualization
#
# This notebook generates a Voronoi diagram based on a 2D lattice of points.
# The lattice points are derived from a basis matrix and a dilation matrix \( K \).
# The visualization helps illustrate the closest regions around each lattice point.

#%% Import Required Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.linalg import companion

# Import functions from LatticeForge
from LatticeForge.lattice_utils import normdet, samplecube, ndgridmat, rasterize, striplattice
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

#%%

R = np.eye(2)/10
xofs = np.array([0.25, 0.25])

points = samplecube(R, xofs=xofs, epsdirub=1);

# %%

#%%
# ## Generate the Dilation Matrix `genK`

def genK(ND, D, coe=0):
    """
    Generate a dilation matrix K.
    
    Parameters:
    - ND: int, Dimension of the matrix.
    - D: int, Determinant of the matrix.q
    - coe: int, Central coefficient (only applies if ND is even).
    
    Returns:
    - K: numpy.ndarray, Dilation matrix.
    """
    if ND % 2:
        cp = np.ones(ND + 1)
        cp[1:ND] = 0
        cp[ND] = -D
    else:
        cp = np.ones(ND + 1)
        cp[1:ND // 2] = 0
        cp[ND // 2 + 1:ND] = 0
        cp[ND] = D
        cp[ND // 2] = coe
        max_coe = int(np.floor(2 * np.sqrt(D)))
        assert abs(coe) <= max_coe, f"Central coefficient coe={coe} exceeds the allowed range ±{max_coe}"
    
    K = companion(cp)
    logging.debug(f"Dilation Matrix K:\n{K}")
    return K

#%%
# ## Generate Bases

def generate_bases(beta=2, N=2, scale=10):
    """
    Generate a random basis R and a dilated basis R * K.
    
    Parameters:
    - beta: int, Determinant of the dilation matrix.
    - N: int, Dimension of the basis.
    - scale: float, Scaling factor for the basis.
    
    Returns:
    - R: numpy.ndarray, Random basis matrix.
    - dilated_basis: numpy.ndarray, Dilated basis matrix.
    """
    R = normdet(np.random.rand(N, N), target_det=1/scale)
    K = genK(N, beta)
    dilated_basis = (R @ K)
    
    logging.debug(f"Original Basis R:\n{R}")
    logging.debug(f"Dilated Basis R * K:\n{dilated_basis}")
    return R, dilated_basis

#%% 
# ## Generate Lattice Points

def generate_lattice_points(basis, eps=1e-3):
    """
    Generate lattice points using the samplecube function.
    
    Parameters:
    - basis: numpy.ndarray, Basis matrix for the lattice.
    - eps: float, Tolerance for sampling.
    
    Returns:
    - points: numpy.ndarray, Generated lattice points.
    """
    points = samplecube(basis, epsdirub=eps)
    logging.debug(f"Generated Points:\n{points}")
    return points

#%%
# ## Plot Voronoi Diagram

def plot_voronoi_with_lattices(beta=2, N=2, eps=1e-3):
    """
    Plot Voronoi diagrams for coarse and fine lattices.
    
    Parameters:
    - beta: int, Determinant of the dilation matrix.
    - N: int, Dimension of the lattice.
    - eps: float, Tolerance for sampling points.
    """
    R, dilated_basis = generate_bases(beta=beta, N=N, scale=23)

    # Normalize R to avoid excessively large values
    # R = normdet(R, 1/23)

    # Generate lattice points
    fine_points = generate_lattice_points(R, eps=eps)
    coarse_points = generate_lattice_points(dilated_basis, eps=eps)

    # Check if points are generated correctly
    if fine_points.size == 0 or coarse_points.size == 0:
        raise ValueError("No points to create Voronoi diagram.")

    # Combine points for Voronoi calculation
    all_points = np.vstack([coarse_points, fine_points])
    vor = Voronoi(all_points)

    # Plot the Voronoi diagram
    fig, ax = plt.subplots(figsize=(8, 8))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='blue', line_width=1, line_alpha=0.6)
    ax.plot(coarse_points[:, 0], coarse_points[:, 1], 'ro', markersize=8, label='Coarse Grid Points')
    ax.plot(fine_points[:, 0], fine_points[:, 1], 'go', markersize=4, label='Fine Grid Points')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Voronoi Diagram with β={beta} Dilation')
    ax.legend()
    ax.grid(True)
    plt.show()

#%%

plot_voronoi_with_lattices(beta=2, N=2, eps=1e-3)

# %%
