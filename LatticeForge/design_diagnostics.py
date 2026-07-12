"""Space-filling diagnostics for finite lattice-design point sets.

These helpers intentionally stay at the paper-facing diagnostic layer.  A
lattice construction determines the points that can be admitted; this module
reads an admitted finite design under ordinary geometric and GP-metric frames.
"""

from itertools import combinations, product

import numpy as np


def metric_transform(points, metric=None, length_scales=None):
    """Return points transformed into the requested metric frame.

    Parameters
    ----------
    points : array-like, shape (n, d)
        Design points.
    metric : array-like, optional
        Positive definite matrix ``M`` used in ``(x-y)^T M (x-y)`` distances.
    length_scales : array-like, optional
        Diagonal GP length scales. Coordinates are transformed as
        ``z_j = x_j / theta_j``.
    """
    pts = _as_points(points)
    if metric is not None and length_scales is not None:
        raise ValueError("Use either metric or length_scales, not both.")
    if length_scales is not None:
        theta = np.asarray(length_scales, dtype=float)
        if theta.shape != (pts.shape[1],):
            raise ValueError("length_scales must have shape (dimension,).")
        if np.any(theta <= 0):
            raise ValueError("length_scales must be positive.")
        return pts / theta
    if metric is not None:
        M = np.asarray(metric, dtype=float)
        if M.shape != (pts.shape[1], pts.shape[1]):
            raise ValueError("metric must have shape (dimension, dimension).")
        L = np.linalg.cholesky(M)
        return pts @ L
    return pts


def metric_gram(basis, metric=None, length_scales=None):
    """Return ``Gamma_M = G.T @ M @ G`` for a lattice basis ``G``.

    This is the first public invariant to test for GP-frame search reduction:
    different basis/rotation presentations that induce the same ``Gamma_M`` are
    candidates for redundant pairwise-metric evaluations, modulo region and
    shift effects.
    """
    G = np.asarray(basis, dtype=float)
    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError("basis must be a square matrix.")
    if metric is not None and length_scales is not None:
        raise ValueError("Use either metric or length_scales, not both.")
    if length_scales is not None:
        theta = np.asarray(length_scales, dtype=float)
        if theta.shape != (G.shape[0],):
            raise ValueError("length_scales must have shape (dimension,).")
        if np.any(theta <= 0):
            raise ValueError("length_scales must be positive.")
        M = np.diag(1.0 / (theta * theta))
    elif metric is not None:
        M = np.asarray(metric, dtype=float)
        if M.shape != G.shape:
            raise ValueError("metric must have the same shape as basis.")
    else:
        M = np.eye(G.shape[0])
    return G.T @ M @ G


def separation_distance(points, metric=None, length_scales=None):
    """Return half the minimum pairwise distance of a finite design."""
    pts = metric_transform(points, metric=metric, length_scales=length_scales)
    distances = _pairwise_distances(pts)
    if distances.size == 0:
        return np.inf
    return float(np.min(distances) / 2.0)


def nearest_pair_distance(points, metric=None, length_scales=None):
    """Return the minimum pairwise distance of a finite design."""
    pts = metric_transform(points, metric=metric, length_scales=length_scales)
    distances = _pairwise_distances(pts)
    if distances.size == 0:
        return np.inf
    return float(np.min(distances))


def fill_distance_grid(
    points,
    bounds=None,
    grid_size=11,
    metric=None,
    length_scales=None,
    max_grid_points=250000,
):
    """Approximate fill distance by a deterministic tensor grid.

    ``bounds`` is a list of ``(lower, upper)`` pairs in the original coordinate
    frame. The grid and design points are both transformed before distances are
    computed.
    """
    pts = _as_points(points)
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if bounds is None:
        bounds = [(0.0, 1.0)] * pts.shape[1]
    bounds = _as_bounds(bounds, pts.shape[1])
    grid_count = grid_size ** pts.shape[1]
    if grid_count > max_grid_points:
        raise ValueError(
            "grid would contain "
            f"{grid_count} points; raise max_grid_points explicitly if intended."
        )
    axes = [np.linspace(lo, hi, grid_size) for lo, hi in bounds]
    grid = np.array(list(product(*axes)), dtype=float)
    pts_metric = metric_transform(pts, metric=metric, length_scales=length_scales)
    grid_metric = metric_transform(grid, metric=metric, length_scales=length_scales)
    return float(np.max(_min_distances_to_design(grid_metric, pts_metric)))


def mesh_ratio_grid(
    points,
    bounds=None,
    grid_size=11,
    metric=None,
    length_scales=None,
    max_grid_points=250000,
):
    """Approximate mesh ratio ``fill_distance / separation_distance``."""
    sep = separation_distance(points, metric=metric, length_scales=length_scales)
    fill = fill_distance_grid(
        points,
        bounds=bounds,
        grid_size=grid_size,
        metric=metric,
        length_scales=length_scales,
        max_grid_points=max_grid_points,
    )
    if sep == 0:
        return np.inf
    return float(fill / sep)


def projected_diagnostics(
    points,
    projection_orders=(1, 2),
    bounds=None,
    grid_size=11,
    max_grid_points=250000,
):
    """Return separation/fill/mesh diagnostics for coordinate projections."""
    pts = _as_points(points)
    dim = pts.shape[1]
    bounds = _as_bounds(bounds or [(0.0, 1.0)] * dim, dim)
    reports = {}
    for order in projection_orders:
        if order < 1 or order > dim:
            continue
        for dims in combinations(range(dim), order):
            proj_points = pts[:, dims]
            proj_bounds = [bounds[j] for j in dims]
            key = ",".join(str(j) for j in dims)
            sep = separation_distance(proj_points)
            fill = fill_distance_grid(
                proj_points,
                bounds=proj_bounds,
                grid_size=grid_size,
                max_grid_points=max_grid_points,
            )
            reports[key] = {
                "dims": dims,
                "separation": sep,
                "fill_distance": fill,
                "mesh_ratio": np.inf if sep == 0 else float(fill / sep),
            }
    return reports


def design_diagnostics(
    points,
    bounds=None,
    grid_size=11,
    projection_orders=(1, 2),
    metric=None,
    length_scales=None,
    max_grid_points=250000,
):
    """Return a compact diagnostics report for a finite design."""
    pts = _as_points(points)
    sep = separation_distance(pts, metric=metric, length_scales=length_scales)
    fill = fill_distance_grid(
        pts,
        bounds=bounds,
        grid_size=grid_size,
        metric=metric,
        length_scales=length_scales,
        max_grid_points=max_grid_points,
    )
    return {
        "n_points": int(pts.shape[0]),
        "dimension": int(pts.shape[1]),
        "separation": sep,
        "nearest_pair_distance": nearest_pair_distance(
            pts, metric=metric, length_scales=length_scales
        ),
        "fill_distance": fill,
        "mesh_ratio": np.inf if sep == 0 else float(fill / sep),
        "projected": projected_diagnostics(
            pts,
            projection_orders=projection_orders,
            bounds=bounds,
            grid_size=grid_size,
            max_grid_points=max_grid_points,
        ),
    }


def _as_points(points):
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n_points, dimension).")
    if pts.shape[0] == 0:
        raise ValueError("points must contain at least one point.")
    return pts


def _as_bounds(bounds, dim):
    if len(bounds) != dim:
        raise ValueError("bounds must contain one (lower, upper) pair per dimension.")
    out = []
    for lo, hi in bounds:
        lo = float(lo)
        hi = float(hi)
        if not lo < hi:
            raise ValueError("each bound must satisfy lower < upper.")
        out.append((lo, hi))
    return out


def _pairwise_distances(points):
    n = points.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    distances = []
    for i in range(n - 1):
        delta = points[i + 1 :] - points[i]
        distances.extend(np.sqrt(np.sum(delta * delta, axis=1)))
    return np.asarray(distances, dtype=float)


def _min_distances_to_design(candidates, design):
    distances = []
    for x in candidates:
        delta = design - x
        distances.append(np.min(np.sqrt(np.sum(delta * delta, axis=1))))
    return np.asarray(distances, dtype=float)
