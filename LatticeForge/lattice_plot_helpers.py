#%% lattice_plot_helpers.py
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D

def showlattice2(R, points=10, marker='o', color='b'):
    """
    Display a 2D lattice for the given basis.

    Parameters:
    - R (np.ndarray): 2x2 lattice basis matrix.
    - points (int): Number of lattice points to display along each direction.
    - marker (str): Marker style for the lattice points.
    - color (str): Color for the lattice points.
    """
    x = np.arange(-points, points + 1)
    grid = np.array(list(product(x, x)))
    lattice_points = grid @ R.T

    plt.figure(figsize=(8, 8))
    plt.plot(lattice_points[:, 0], lattice_points[:, 1], marker, color=color)
    
    # Plot basis vectors
    origin = np.array([0, 0])
    plt.quiver(*origin, *R[:, 0], angles='xy', scale_units='xy', scale=1, color='r', label='Basis Vector 1')
    plt.quiver(*origin, *R[:, 1], angles='xy', scale_units='xy', scale=1, color='g', label='Basis Vector 2')
    
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title('2D Lattice')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def distancemap(R, ext=5):
    """
    Generate a distance map for a 2D lattice.

    Parameters:
    - R (np.ndarray): 2x2 lattice basis matrix.
    - ext (int): Range for the distance map.
    """
    iaxis = np.arange(-ext, ext + 1)
    X1, X2 = np.meshgrid(iaxis, iaxis)
    Z2 = np.vstack([X1.ravel(), X2.ravel()])
    L = R @ Z2
    dlist = np.sum(L**2, axis=0)
    dimg = dlist.reshape(2 * ext + 1, 2 * ext + 1)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log10(dimg + 0.1), extent=[-ext, ext, -ext, ext], origin='lower', cmap='hot')
    plt.colorbar(label='Log Distance')
    plt.title('Distance Map for 2D Lattice')
    plt.xlabel('Index X')
    plt.ylabel('Index Y')
    plt.grid(True)
    plt.show()

def show_rayleigh2(R, N=200):
    """
    Create a spherical plot of the Rayleigh quotient of R'*R.

    Parameters:
    - R (np.ndarray): 2x2 lattice basis matrix.
    - N (int): Number of angles to sample.
    """
    RR = R.T @ R
    phi = np.linspace(0, 2 * np.pi, N)
    x = np.vstack([np.cos(phi), np.sin(phi)])
    rhox = np.sum(x * (RR @ x), axis=0)
    
    rpts = R @ x * rhox
    
    plt.figure(figsize=(8, 8))
    plt.plot(rpts[0, :], rpts[1, :], 'b-')
    plt.title('Rayleigh Quotient Spherical Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def fig_latcmp(cr, pr):
    """
    Compare lattices via a plot of their covering vs. packing radii.

    Parameters:
    - cr (list or np.ndarray): List of covering radii.
    - pr (list or np.ndarray): List of packing radii.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(cr, pr, c='b', marker='+')
    plt.plot([0, max(cr)], [0, max(pr)], 'k--', label='Geometric Bound')
    plt.title('Lattice Comparison: Covering vs. Packing Radii')
    plt.xlabel('Covering Radius')
    plt.ylabel('Packing Radius')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def showlattices2(Rlist, points=10, marker='o', color='b'):
    """
    Show lattices for each basis in a list of 2D lattice basis matrices.

    Parameters:
    - Rlist (list of np.ndarray): List of 2x2 lattice basis matrices.
    - points (int): Number of lattice points to display along each direction.
    - marker (str): Marker style for the lattice points.
    - color (str): Color for the lattice points.
    """
    n = len(Rlist)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()
    
    for i, R in enumerate(Rlist):
        x = np.arange(-points, points + 1)
        grid = np.array(list(product(x, x)))
        lattice_points = grid @ R.T
        
        axes[i].plot(lattice_points[:, 0], lattice_points[:, 1], marker, color=color)
        origin = np.array([0, 0])
        axes[i].quiver(*origin, *R[:, 0], angles='xy', scale_units='xy', scale=1, color='r')
        axes[i].quiver(*origin, *R[:, 1], angles='xy', scale_units='xy', scale=1, color='g')
        
        axes[i].grid(True)
        axes[i].set_aspect('equal', adjustable='box')
        axes[i].set_title(f'Lattice {i + 1}')
    
    plt.tight_layout()
    plt.show()
