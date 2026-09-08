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


def _integer_grid_chunks(lower, upper, chunk_size):
    """Yield a Cartesian integer box without materializing coordinate grids.

    The order is the same as ``ndgridmat`` with ``indexing='ij'``: the last
    coordinate changes fastest.  Keeping this iterator private lets
    ``samplecube`` honour its existing memory contract without enlarging the
    public construction API.
    """
    lower = np.asarray(lower, dtype=np.int64)
    upper = np.asarray(upper, dtype=np.int64)
    lengths = upper - lower + 1
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("lower and upper must be one-dimensional and aligned")
    if np.any(lengths <= 0):
        return

    total = int(np.prod(lengths, dtype=object))
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        residual = np.arange(start, stop, dtype=np.int64)
        grid = np.empty((lower.size, stop - start), dtype=np.int64)
        for axis in range(lower.size - 1, -1, -1):
            grid[axis] = lower[axis] + residual % lengths[axis]
            residual //= lengths[axis]
        yield grid


def _candidate_chunk_size(dimension, maxmemMB):
    """Conservatively translate a working-memory budget into a row count."""
    if not np.isscalar(maxmemMB) or not np.isfinite(maxmemMB) or maxmemMB <= 0:
        raise ValueError("maxmemMB must be a positive finite number")

    # One address row and one transformed point use 16*d bytes.  The larger
    # allowance covers matrix-multiply output, the linear-index work vector,
    # and per-coordinate inclusion masks.  Retained points are normally much
    # smaller than the enclosing address box and are accumulated separately.
    bytes_per_candidate = 24 * dimension + 32
    return max(1, int(maxmemMB * 1024 * 1024 // bytes_per_candidate))


def _unit_cube_address_bounds(R_inv, xofs):
    """Return exact coordinate bounds for the inverse image of ``[0, 1]^d``.

    A linear functional reaches its extrema on a box by choosing each endpoint
    according to the sign of its coefficient.  This avoids constructing all
    ``2**d`` cube corners while producing the same enclosing address box.
    """
    shifted_origin = -(R_inv @ xofs)
    lower = shifted_origin + np.minimum(R_inv, 0.0).sum(axis=1)
    upper = shifted_origin + np.maximum(R_inv, 0.0).sum(axis=1)
    return np.floor(lower).astype(np.int64), np.ceil(upper).astype(np.int64)


def _incremental_cube_points(
    R, xofs, lower, upper, eps, epsdirub, maxmemMB, diagnostics=None
):
    """Enumerate a bounded lattice by pruning incomplete integer addresses.

    This recovers the governing idea of the original MATLAB ``rasterize.m``:
    extend an integer address one coordinate at a time, retain only prefixes
    whose lower-dimensional shadow can still meet the target cube, and perform
    an exact membership test after the final coordinate.  The prefix test here
    uses conservative interval enclosures.  It may retain unnecessary prefixes
    but cannot reject a prefix of an in-cube point (up to the declared numeric
    tolerance).
    """
    dimension = R.shape[0]
    lengths = upper - lower + 1

    # Start with columns that have large geometric effect relative to their
    # branching count.  This is only a traversal order; it cannot change the
    # resulting lattice point set.
    effect = np.linalg.norm(R, axis=0) * np.maximum(lengths - 1, 0) / lengths
    order = np.argsort(-effect, kind="stable")

    # Suffix enclosures for every not-yet-chosen address coordinate.
    suffix_lower = np.zeros((dimension + 1, dimension), dtype=float)
    suffix_upper = np.zeros((dimension + 1, dimension), dtype=float)
    for depth in range(dimension - 1, -1, -1):
        axis = order[depth]
        at_lower = R[:, axis] * lower[axis]
        at_upper = R[:, axis] * upper[axis]
        suffix_lower[depth] = (
            suffix_lower[depth + 1] + np.minimum(at_lower, at_upper)
        )
        suffix_upper[depth] = (
            suffix_upper[depth + 1] + np.maximum(at_lower, at_upper)
        )

    cube_lower = -eps
    cube_upper = 1.0 + epsdirub * eps
    numeric_guard = max(
        1.0e-12,
        32.0 * np.finfo(float).eps
        * (1.0 + np.max(np.abs(xofs)) + np.max(np.abs(R)) * np.max(np.abs(upper))),
    )
    max_states = _candidate_chunk_size(dimension, maxmemMB)
    retained = []
    considered_by_depth = [0] * dimension
    retained_by_depth = [0] * dimension
    peak_batch_states = 0

    def visit(addresses, partial_points, depth):
        nonlocal peak_batch_states
        axis = order[depth]
        values = np.arange(lower[axis], upper[axis] + 1, dtype=np.int64)
        parents_per_batch = max(1, max_states // len(values))

        for start in range(0, len(addresses), parents_per_batch):
            stop = min(start + parents_per_batch, len(addresses))
            parent_addresses = addresses[start:stop]
            parent_points = partial_points[start:stop]
            parent_count = len(parent_addresses)

            repeated_values = np.tile(values, parent_count)
            child_addresses = np.repeat(parent_addresses, len(values), axis=0)
            child_addresses[:, axis] = repeated_values
            child_points = (
                np.repeat(parent_points, len(values), axis=0)
                + repeated_values[:, None] * R[:, axis]
            )
            considered_by_depth[depth] += len(child_addresses)
            peak_batch_states = max(peak_batch_states, len(child_addresses))

            possible_lower = child_points + suffix_lower[depth + 1]
            possible_upper = child_points + suffix_upper[depth + 1]
            possible = np.all(
                (possible_upper >= cube_lower - numeric_guard)
                & (possible_lower <= cube_upper + numeric_guard),
                axis=1,
            )
            if not np.any(possible):
                continue

            child_addresses = child_addresses[possible]
            child_points = child_points[possible]
            retained_by_depth[depth] += len(child_addresses)
            if depth + 1 < dimension:
                visit(child_addresses, child_points, depth + 1)
            else:
                # Recompute in one matrix product so final coordinates do not
                # depend on the prefix traversal or its summation order.
                points = (R @ child_addresses.T).T + xofs
                inside = np.ones(points.shape[0], dtype=bool)
                for coordinate in range(dimension):
                    inside &= points[:, coordinate] >= cube_lower
                    inside &= points[:, coordinate] <= cube_upper
                if np.any(inside):
                    retained.append(points[inside])

    visit(
        np.zeros((1, dimension), dtype=np.int64),
        np.asarray(xofs, dtype=float)[None, :],
        0,
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "traversal_order": [int(axis) for axis in order],
                "considered_by_depth": considered_by_depth,
                "retained_by_depth": retained_by_depth,
                "peak_batch_states": peak_batch_states,
                "memory_budget_MiB": float(maxmemMB),
            }
        )
    return (
        np.concatenate(retained, axis=0)
        if retained
        else np.empty((0, dimension), dtype=float)
    )


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
    - maxmemMB (float, optional): Approximate working-memory budget in MiB for
      address traversal. The returned point matrix itself is not truncated.

    Returns:
    - np.ndarray: Sampled lattice points within the hypercube.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square matrix")
    N = R.shape[0]
    if xofs is None:
        xofs = np.zeros(N)
    else:
        xofs = np.asarray(xofs, dtype=float)
        if xofs.shape != (N,):
            raise ValueError("xofs must have shape (dimension,)")

    # Compute the inverse of R to determine the bounds in the indexing space
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        raise ValueError("R is singular or not invertible")

    # Determine the inverse-image address box by interval arithmetic.  This is
    # equivalent to transforming all cube corners but does not allocate 2**N
    # corner rows.
    LB, UB = _unit_cube_address_bounds(R_inv, xofs)

    logging.debug(f"Calculated Lower Bound (LB):\n{LB}")
    logging.debug(f"Calculated Upper Bound (UB):\n{UB}")

    lengths = UB - LB + 1
    candidate_count = int(np.prod(lengths, dtype=object))
    expected_count = 1.0 / abs(np.linalg.det(R))
    use_incremental = N > 7 or candidate_count > max(4096, 8 * expected_count)

    if use_incremental:
        logging.debug(
            "Using incremental shadow-pruned traversal for %d address candidates",
            candidate_count,
        )
        filtered_points = _incremental_cube_points(
            R, xofs, LB, UB, eps, epsdirub, maxmemMB
        )
    else:
        # Traverse the comprehensive address box in bounded chunks.  The
        # former implementation created N full mesh grids plus a stacked grid,
        # so its peak working set grew by roughly N times the already-large
        # address matrix. ``maxmemMB`` now governs the chunk working set.
        chunk_size = _candidate_chunk_size(N, maxmemMB)
        retained = []
        for grid in _integer_grid_chunks(LB, UB, chunk_size):
            points = (R @ grid).T + xofs
            inside = np.ones(points.shape[0], dtype=bool)
            for axis in range(N):
                inside &= points[:, axis] >= -eps
                inside &= points[:, axis] <= 1 + epsdirub * eps
            if np.any(inside):
                retained.append(points[inside])

        filtered_points = (
            np.concatenate(retained, axis=0)
            if retained
            else np.empty((0, N), dtype=float)
        )

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
