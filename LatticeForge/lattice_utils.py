#%% lattice_utils.py
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

import logging

def normdet(R, target_det=1.0):
    """
    Normalize the determinant of a matrix to a target value.

    Parameters:
    - R (np.ndarray): Input matrix.
    - target_det (float): Desired determinant value.

    Returns:
    - np.ndarray: Matrix with normalized determinant.
    """
    current_det = np.abs(np.linalg.det(R))
    if current_det < 1e-12:
        raise ValueError("Determinant is too close to zero; cannot normalize.")
    scaling_factor = (target_det / current_det) ** (1 / R.shape[0])
    return R * scaling_factor

def striplattice(R):
    """
    Reduce a lattice matrix using the striplattice algorithm with improved stability.

    Parameters:
    - R (np.ndarray): Lattice basis matrix.

    Returns:
    - np.ndarray: Reduced lattice matrix.
    """
    R = R.astype(np.float64)
    N = R.shape[0]
    M = np.eye(N, dtype=np.float64)
    
    max_iterations = N * 5  # Reduced number of iterations
    max_value_threshold = 5.0  # Reduced threshold for matrix values
    
    for _ in range(max_iterations):
        S = np.linalg.pinv(R.T @ R)  # Use pseudo-inverse for stability
        U = np.round(S / np.diag(S)[:, None]) - np.eye(N)
        max_idx = np.unravel_index(np.argmax(np.abs(U)), U.shape)

        if U[max_idx] == 0:
            break

        # More conservative update
        update_scale = 0.5  # Limit the magnitude of updates
        update = M @ (U[:, max_idx[1]] * update_scale)
        
        M[:, max_idx[1]] += update
        R[:, max_idx[1]] += R @ (U[:, max_idx[1]] * update_scale)

        # Stricter value capping
        R = np.clip(R, -max_value_threshold, max_value_threshold)

        # Early exit if values are sufficiently stable
        if np.all(np.abs(U) < 1e-10):
            break

    return R

def ndgridmat(ranges):
    """
    Create points in an N-dimensional coordinate grid.

    Parameters:
    - ranges (list of arrays): List of 1D coordinate arrays for each dimension.

    Returns:
    - np.ndarray: Points in the N-dimensional grid.
    """
    grids = np.meshgrid(*ranges, indexing='ij')
    return np.vstack([g.ravel() for g in grids])


def rasterize(R, xofs=None, eps=1e-7, minbvol=False, sortit=False):
    """
    Generate lattice points within a unit cube, with optimizations for high-dimensional cases using bounding epiped.

    Parameters:
    - R (np.ndarray): Lattice basis matrix.
    - xofs (np.ndarray, optional): Shift vector. Default is no shift.
    - eps (float, optional): Tolerance for boundary inclusion.
    - minbvol (bool, optional): If True, use minimal bounding volume optimization.
    - sortit (bool, optional): Sort basis vectors for efficiency.

    Returns:
    - np.ndarray: Rasterized lattice points.
    """
    N = R.shape[0]
    if xofs is None:
        xofs = np.zeros(N)

    # Sort basis vectors for efficiency if required
    if sortit and N > 1:
        sort_indices = np.argsort(np.sum(R * R, axis=1))
        R = R[sort_indices, :]
        xofs = xofs[sort_indices]

    # Compute the bounding epiped
    bbox = boundingepiped(R)
    lb, ub = bbox
    lb_flat = lb.flatten()
    ub_flat = ub.flatten()

    # Generate grid points within the bounding box
    ranges = [np.arange(int(np.floor(lb_flat[i] - eps)), int(np.ceil(ub_flat[i] + eps)) + 1) for i in range(N)]
    grid = ndgridmat(ranges)

    # Transform grid points back to the original space and apply the shift
    points = (R @ grid) + xofs[:, None]

    # Filter points that lie within the unit hypercube
    inside = np.all((points >= -eps) & (points <= 1 + eps), axis=0)

    return points[:, inside].T

def samplecube(R, xofs=None, eps=1e-5, epsdirub=1, maxmemMB=128):
    """
    Generate lattice points within the dynamically determined hypercube region.

    Parameters:
    - R (np.ndarray): Basis matrix of the lattice.
    - xofs (np.ndarray, optional): Shift vector for the lattice. Default is no shift.
    - eps (float, optional): Tolerance for including boundary points.
    - epsdirub (int, optional): Direction of boundary tolerance at the upper bound.
    - maxmemMB (int, optional): Maximum allowable memory for candidate point sets in MB.

    Returns:
    - np.ndarray: Sampled lattice points within the hypercube.
    """
    N = R.shape[0]
    if xofs is None:
        xofs = np.zeros(N)

    # Compute the inverse of R to determine the bounds in the indexing space
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        raise ValueError("R is singular or not invertible")

    # Define the unit hypercube corners in the target space
    unit_corners = np.array(np.meshgrid(*([[0, 1]] * N))).T.reshape(-1, N)

    # Map the unit hypercube corners to the indexing space
    bbox_corners = (unit_corners - xofs) @ R_inv.T

    # Determine LB (lower bound) and UB (upper bound) by rounding down and up to the nearest integers
    LB = np.floor(np.min(bbox_corners, axis=0)).astype(int)
    UB = np.ceil(np.max(bbox_corners, axis=0)).astype(int)

    logging.debug(f"Bounding Box Corners in Indexing Space:\n{bbox_corners}")
    logging.debug(f"Calculated Lower Bound (LB):\n{LB}")
    logging.debug(f"Calculated Upper Bound (UB):\n{UB}")

    # Generate comprehensive grid of points within the determined bounds
    ranges = [np.arange(LB[i], UB[i] + 1) for i in range(N)]
    grid = ndgridmat(ranges)

    # Transform grid points back to the target space
    points = (R @ grid).T + xofs

    # Robust filtering of points to ensure they lie within the unit hypercube
    inside = np.all(
        (points >= -eps) & (points <= 1 + epsdirub * eps),
        axis=1
    )

    filtered_points = points[inside]

    # Ensure unique points
    filtered_points = np.unique(filtered_points, axis=0)

    return filtered_points

from itertools import product

def sampleregion(R, isin, xofs=None, extendring=1):
    """
    Construct lattice points inside a region defined by the `isin` function.

    Parameters:
    - R (np.ndarray): Basis matrix.
    - isin (callable): Function returning True if a point is in the region.
    - xofs (np.ndarray, optional): Offset index for the region.
    - extendring (int or np.ndarray, optional): If > 0, include neighbors in the growing kernel.

    Returns:
    - np.ndarray: Points inside the region.
    """
    if xofs is None:
        xofs = np.zeros(R.shape[0])
    
    ndim = R.shape[0]
    
    if isinstance(extendring, np.ndarray):
        K = extendring
        extendring = 0
    else:
        K = np.eye(ndim)
    
    vc = relevant_vectors(R)
    
    points = []
    queue = [xofs]
    visited = set()
    
    while queue:
        current = queue.pop(0)
        if tuple(current) in visited:
            continue
        visited.add(tuple(current))
        
        if isin(current @ R.T):
            points.append(current @ R.T)
            for neighbor in vc:
                new_point = current + neighbor
                if tuple(new_point) not in visited:
                    queue.append(new_point)
    
    return np.array(points)

def relevant_vectors(R):
    """
    Generate relevant vectors for the given basis matrix R.
    """
    ndim = R.shape[0]
    directions = np.array(list(product([-1, 0, 1], repeat=ndim)))
    directions = directions[np.any(directions != 0, axis=1)]
    return directions

def boundingepiped(Q, optimize_det=False):
    """
    Compute the bounding parallelepiped for a given matrix Q.

    Parameters:
    - Q (np.ndarray): Input matrix.
    - optimize_det (bool, optional): Whether to optimize determinant.

    Returns:
    - Tuple[np.ndarray, np.ndarray]: Bounding parallelepiped matrix and offset vector.
    """
    di, ndim = Q.shape
    if di == ndim:
        return Q, np.zeros(di)
    
    S = np.array(list(product([-1, 1], repeat=ndim)))
    X = Q @ S.T
    
    xofs = np.min(X, axis=1)
    Qb = np.max(X, axis=1) - xofs
    
    return Qb, xofs

def reduce_relevant(R):
    """
    Find a reduced basis for a lattice using relevant vectors.

    Parameters:
    - R (np.ndarray): Lattice basis matrix.

    Returns:
    - np.ndarray: Reduced basis matrix.
    """
    ndim = R.shape[0]
    vc = relevant_vectors(R)
    vr = vc @ R
    dist = np.sum(vr ** 2, axis=1)
    
    sorted_indices = np.argsort(dist)
    vc = vc[sorted_indices]
    vr = vr[sorted_indices]
    
    B = np.zeros((ndim, ndim))
    B[0] = vr[0]
    
    for i in range(1, ndim):
        Br = B[:i]
        pinvB = np.linalg.pinv(Br)
        residuals = vr - vr @ pinvB @ Br
        non_zero = np.where(np.linalg.norm(residuals, axis=1) > 1e-6)[0]
        B[i] = vr[non_zero[0]]
    
    return B.T

