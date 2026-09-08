"""Memory-bounded construction of a lattice design in the unit cube.

This module implements the bounded-region operation stated in Algorithm
``makeDesign`` of the RotGrid manuscript.  It determines a finite box of
integer lattice addresses, traverses that box in bounded batches, and retains
exactly the points admitted by the requested half-open or tolerance-adjusted
cube.  For large boxes it extends addresses one coordinate at a time and
prunes only prefixes whose remaining coordinate enclosure cannot meet the
cube.

The module owns enumeration mechanics only.  It does not choose a lattice,
rotation, shift, refinement path, or statistical criterion.
"""

from __future__ import annotations

from collections.abc import Iterator, MutableMapping
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass(frozen=True)
class CubePointBatch:
    """One bounded batch of integer addresses and their physical points."""

    addresses: np.ndarray
    points: np.ndarray


@dataclass
class CubeTraversalDiagnostics:
    """Observable work performed by one complete cube traversal."""

    strategy: str = ""
    address_box_count: int = 0
    traversal_order: list[int] = field(default_factory=list)
    considered_by_depth: list[int] = field(default_factory=list)
    retained_by_depth: list[int] = field(default_factory=list)
    peak_batch_states: int = 0
    emitted_point_count: int = 0
    memory_budget_MiB: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready copy of the completed diagnostics."""
        return asdict(self)


def _candidate_chunk_size(dimension: int, max_memory_mib: float) -> int:
    """Conservatively translate a working-memory budget into a row count."""
    if (
        not np.isscalar(max_memory_mib)
        or not np.isfinite(max_memory_mib)
        or max_memory_mib <= 0
    ):
        raise ValueError("max_memory_mib must be a positive finite number")

    # One address row and one transformed point use 16*d bytes.  The larger
    # allowance covers matrix-multiply output, mixed-radix work, and masks.
    bytes_per_candidate = 24 * dimension + 32
    return max(
        1, int(float(max_memory_mib) * 1024 * 1024 // bytes_per_candidate)
    )


def _integer_grid_chunks(
    lower: np.ndarray, upper: np.ndarray, chunk_size: int
) -> Iterator[np.ndarray]:
    """Yield a Cartesian integer box without constructing coordinate grids."""
    lower = np.asarray(lower, dtype=np.int64)
    upper = np.asarray(upper, dtype=np.int64)
    if lower.ndim != 1 or upper.shape != lower.shape:
        raise ValueError("lower and upper must be one-dimensional and aligned")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    lengths = upper - lower + 1
    if np.any(lengths <= 0):
        return

    total = int(np.prod(lengths, dtype=object))
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        residual = np.arange(start, stop, dtype=np.int64)
        addresses = np.empty((lower.size, stop - start), dtype=np.int64)
        for axis in range(lower.size - 1, -1, -1):
            addresses[axis] = lower[axis] + residual % lengths[axis]
            residual //= lengths[axis]
        yield addresses


def _unit_cube_address_bounds(
    inverse_basis: np.ndarray,
    shift: np.ndarray,
    cube_lower: float | np.ndarray = 0.0,
    cube_upper: float | np.ndarray = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bound the inverse image of an axis-aligned cube coordinatewise."""
    inverse_basis = np.asarray(inverse_basis, dtype=float)
    shift = np.asarray(shift, dtype=float)
    dimension = inverse_basis.shape[0]
    lower = np.broadcast_to(np.asarray(cube_lower, dtype=float), (dimension,))
    upper = np.broadcast_to(np.asarray(cube_upper, dtype=float), (dimension,))
    if np.any(lower > upper):
        raise ValueError("cube_lower must not exceed cube_upper")

    shifted_origin = inverse_basis @ (lower - shift)
    contributions = inverse_basis * (upper - lower)[None, :]
    preimage_lower = shifted_origin + np.minimum(contributions, 0.0).sum(axis=1)
    preimage_upper = shifted_origin + np.maximum(contributions, 0.0).sum(axis=1)
    return (
        np.floor(preimage_lower).astype(np.int64),
        np.ceil(preimage_upper).astype(np.int64),
    )


def unit_cube_address_bounds(
    basis: np.ndarray,
    shift: np.ndarray | None = None,
    *,
    eps: float = 1.0e-7,
    upper_tolerance_direction: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the finite address box used for a unit-cube design.

    The accepted physical interval is ``[-eps, 1 + direction*eps]`` in every
    coordinate.  Use ``direction=-1`` for the manuscript's half-open cube
    convention or ``direction=1`` to include an upper tolerance band.
    """
    basis, shift = _validated_basis_and_shift(basis, shift)
    inverse_basis = np.linalg.inv(basis)
    return _unit_cube_address_bounds(
        inverse_basis,
        shift,
        cube_lower=-eps,
        cube_upper=1.0 + upper_tolerance_direction * eps,
    )


def _validated_basis_and_shift(
    basis: np.ndarray, shift: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    basis = np.asarray(basis, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
        raise ValueError("basis must be a square matrix")
    dimension = basis.shape[0]
    if dimension == 0:
        raise ValueError("basis must have positive dimension")
    if not np.all(np.isfinite(basis)):
        raise ValueError("basis must contain only finite values")
    try:
        np.linalg.inv(basis)
    except np.linalg.LinAlgError as error:
        raise ValueError("basis is singular or not invertible") from error

    if shift is None:
        prepared_shift = np.zeros(dimension, dtype=float)
    else:
        prepared_shift = np.asarray(shift, dtype=float)
        if prepared_shift.shape != (dimension,):
            raise ValueError("shift must have shape (dimension,)")
        if not np.all(np.isfinite(prepared_shift)):
            raise ValueError("shift must contain only finite values")
    return basis, prepared_shift


def _inside_cube(
    points: np.ndarray, cube_lower: float, cube_upper: float
) -> np.ndarray:
    inside = np.ones(points.shape[0], dtype=bool)
    for coordinate in range(points.shape[1]):
        inside &= points[:, coordinate] >= cube_lower
        inside &= points[:, coordinate] <= cube_upper
    return inside


def _iter_incremental_batches(
    basis: np.ndarray,
    shift: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    cube_lower: float,
    cube_upper: float,
    max_memory_mib: float,
    diagnostics: CubeTraversalDiagnostics,
) -> Iterator[CubePointBatch]:
    """Extend addresses by one coordinate and conservatively prune prefixes."""
    dimension = basis.shape[0]
    lengths = upper - lower + 1

    # This order changes work only.  It cannot change the accepted address set.
    effect = np.linalg.norm(basis, axis=0) * np.maximum(lengths - 1, 0) / lengths
    order = np.argsort(-effect, kind="stable")
    diagnostics.traversal_order = [int(axis) for axis in order]
    diagnostics.considered_by_depth = [0] * dimension
    diagnostics.retained_by_depth = [0] * dimension

    suffix_lower = np.zeros((dimension + 1, dimension), dtype=float)
    suffix_upper = np.zeros((dimension + 1, dimension), dtype=float)
    for depth in range(dimension - 1, -1, -1):
        axis = order[depth]
        at_lower = basis[:, axis] * lower[axis]
        at_upper = basis[:, axis] * upper[axis]
        suffix_lower[depth] = (
            suffix_lower[depth + 1] + np.minimum(at_lower, at_upper)
        )
        suffix_upper[depth] = (
            suffix_upper[depth + 1] + np.maximum(at_lower, at_upper)
        )

    numeric_guard = max(
        1.0e-12,
        32.0
        * np.finfo(float).eps
        * (
            1.0
            + np.max(np.abs(shift))
            + np.max(np.abs(basis)) * np.max(np.abs(upper))
        ),
    )
    max_states = _candidate_chunk_size(dimension, max_memory_mib)

    def visit(
        addresses: np.ndarray, partial_points: np.ndarray, depth: int
    ) -> Iterator[CubePointBatch]:
        axis = order[depth]
        values = np.arange(lower[axis], upper[axis] + 1, dtype=np.int64)
        for value_start in range(0, len(values), max_states):
            value_stop = min(value_start + max_states, len(values))
            value_batch = values[value_start:value_stop]
            parents_per_batch = max(1, max_states // len(value_batch))

            for start in range(0, len(addresses), parents_per_batch):
                stop = min(start + parents_per_batch, len(addresses))
                parent_addresses = addresses[start:stop]
                parent_points = partial_points[start:stop]
                parent_count = len(parent_addresses)

                repeated_values = np.tile(value_batch, parent_count)
                child_addresses = np.repeat(
                    parent_addresses, len(value_batch), axis=0
                )
                child_addresses[:, axis] = repeated_values
                child_points = (
                    np.repeat(parent_points, len(value_batch), axis=0)
                    + repeated_values[:, None] * basis[:, axis]
                )
                diagnostics.considered_by_depth[depth] += len(child_addresses)
                diagnostics.peak_batch_states = max(
                    diagnostics.peak_batch_states, len(child_addresses)
                )

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
                diagnostics.retained_by_depth[depth] += len(child_addresses)
                if depth + 1 < dimension:
                    yield from visit(child_addresses, child_points, depth + 1)
                    continue

                # Recompute once in the declared basis so returned coordinates
                # do not depend on the prefix summation order.
                points = (basis @ child_addresses.T).T + shift
                inside = _inside_cube(points, cube_lower, cube_upper)
                if np.any(inside):
                    batch = CubePointBatch(
                        addresses=child_addresses[inside], points=points[inside]
                    )
                    diagnostics.emitted_point_count += len(batch.points)
                    yield batch

    yield from visit(
        np.zeros((1, dimension), dtype=np.int64),
        shift[None, :],
        0,
    )


def iter_unit_cube_batches(
    basis: np.ndarray,
    shift: np.ndarray | None = None,
    *,
    eps: float = 1.0e-7,
    upper_tolerance_direction: int = -1,
    max_memory_mib: float = 128.0,
    diagnostics: CubeTraversalDiagnostics | None = None,
) -> Iterator[CubePointBatch]:
    """Yield bounded lattice addresses and points without storing the result.

    Each yielded batch satisfies ``points = addresses @ basis.T + shift`` and
    lies inside the declared tolerance-adjusted unit cube.  Consuming the
    iterator to exhaustion completes ``diagnostics`` when one is supplied.
    """
    basis, shift = _validated_basis_and_shift(basis, shift)
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite nonnegative number")
    if upper_tolerance_direction not in (-1, 1):
        raise ValueError("upper_tolerance_direction must be -1 or 1")
    _candidate_chunk_size(basis.shape[0], max_memory_mib)

    cube_lower = -float(eps)
    cube_upper = 1.0 + upper_tolerance_direction * float(eps)
    if cube_lower > cube_upper:
        raise ValueError("eps leaves an empty unit-cube acceptance interval")
    inverse_basis = np.linalg.inv(basis)
    lower, upper = _unit_cube_address_bounds(
        inverse_basis, shift, cube_lower, cube_upper
    )
    lengths = upper - lower + 1
    address_box_count = int(np.prod(lengths, dtype=object))
    expected_count = 1.0 / abs(np.linalg.det(basis))
    use_incremental = (
        basis.shape[0] > 7
        or address_box_count > max(4096, 8 * expected_count)
    )

    if diagnostics is None:
        diagnostics = CubeTraversalDiagnostics()
    diagnostics.strategy = (
        "dimension_incremental" if use_incremental else "cartesian_chunks"
    )
    diagnostics.address_box_count = address_box_count
    diagnostics.memory_budget_MiB = float(max_memory_mib)

    if use_incremental:
        yield from _iter_incremental_batches(
            basis,
            shift,
            lower,
            upper,
            cube_lower,
            cube_upper,
            max_memory_mib,
            diagnostics,
        )
        return

    diagnostics.traversal_order = list(range(basis.shape[0]))
    chunk_size = _candidate_chunk_size(basis.shape[0], max_memory_mib)
    for address_columns in _integer_grid_chunks(lower, upper, chunk_size):
        addresses = address_columns.T
        diagnostics.peak_batch_states = max(
            diagnostics.peak_batch_states, len(addresses)
        )
        points = addresses @ basis.T + shift
        inside = _inside_cube(points, cube_lower, cube_upper)
        if np.any(inside):
            batch = CubePointBatch(
                addresses=addresses[inside], points=points[inside]
            )
            diagnostics.emitted_point_count += len(batch.points)
            yield batch


def sample_unit_cube(
    basis: np.ndarray,
    shift: np.ndarray | None = None,
    *,
    eps: float = 1.0e-7,
    upper_tolerance_direction: int = -1,
    max_memory_mib: float = 128.0,
) -> np.ndarray:
    """Materialize the unit-cube batches as a conventional point matrix."""
    prepared_basis, _ = _validated_basis_and_shift(basis, shift)
    batches = list(
        iter_unit_cube_batches(
            prepared_basis,
            shift,
            eps=eps,
            upper_tolerance_direction=upper_tolerance_direction,
            max_memory_mib=max_memory_mib,
        )
    )
    if not batches:
        return np.empty((0, prepared_basis.shape[0]), dtype=float)
    return np.unique(np.concatenate([batch.points for batch in batches]), axis=0)


def _incremental_cube_points(
    basis: np.ndarray,
    shift: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eps: float,
    upper_tolerance_direction: int,
    maxmemMB: float,
    diagnostics: MutableMapping[str, object] | None = None,
) -> np.ndarray:
    """Compatibility bridge for the pre-module private implementation helper."""
    prepared_basis, prepared_shift = _validated_basis_and_shift(basis, shift)
    record = CubeTraversalDiagnostics(
        strategy="dimension_incremental",
        address_box_count=int(np.prod(upper - lower + 1, dtype=object)),
        memory_budget_MiB=float(maxmemMB),
    )
    batches = list(
        _iter_incremental_batches(
            prepared_basis,
            prepared_shift,
            np.asarray(lower, dtype=np.int64),
            np.asarray(upper, dtype=np.int64),
            -float(eps),
            1.0 + upper_tolerance_direction * float(eps),
            maxmemMB,
            record,
        )
    )
    if diagnostics is not None:
        diagnostics.update(record.to_dict())
    if not batches:
        return np.empty((0, prepared_basis.shape[0]), dtype=float)
    return np.concatenate([batch.points for batch in batches], axis=0)


__all__ = [
    "CubePointBatch",
    "CubeTraversalDiagnostics",
    "iter_unit_cube_batches",
    "sample_unit_cube",
    "unit_cube_address_bounds",
]
