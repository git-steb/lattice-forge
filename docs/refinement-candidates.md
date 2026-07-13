# Refinement Candidates

Date: 2026-07-12
Status: executable construction surface

This note records the first direct construction route for admissible sequential
lattice-design candidates in LatticeForge.

## Construction

Let `G` be a lattice basis and let `K` be an invertible integer dilation matrix.
The refined lattice has basis

```text
G K^{-1}.
```

It decomposes into finitely many cosets over the coarse lattice:

```text
G K^{-1} Z^d = union_c (G Z^d + G K^{-1} c),
```

where `c` ranges over representatives of `Z^d / K Z^d`.  The number of
representatives is `abs(det K)`.  The zero representative gives the old coarse
lattice; the other representatives give the new refinement layers.

The public constructor is:

```python
from LatticeForge.refinement_candidates import refinement_candidate_points

candidates = refinement_candidate_points(G, K)
```

For `G = I` and `K = 2I` in two dimensions, the non-base cosets generate the
mid-edge and center points of the one-step dyadic refinement inside the unit
square.

## Why This Matters

The earlier code could evaluate point sets and rank candidate additions, but
the candidate set itself could still be hand supplied or obtained through a
broader search.  The coset constructor changes that boundary:

```text
integer dilation K -> admissible refinement offsets -> finite candidate set
```

Packing distance, fill distance, mesh ratio, projection behavior, and
Gaussian-process metric-frame diagnostics can then be read after construction.
They do not have to invent the admissible family.

This is closer in spirit to low-discrepancy constructions such as Halton or
Sobol sequences than to unconstrained point-placement optimization: a lawful
addressing rule creates the next possible continuations, and criteria then
inspect those continuations.

## Current Scope

The implementation currently covers one-step unit-cube refinement:

- `coset_representatives(K)` enumerates integer representatives of
  `Z^d / K Z^d`;
- `refinement_offsets(G, K)` computes the non-base offsets `G K^{-1} c`;
- `refinement_candidate_points(G, K, ...)` samples those offsets in the unit
  cube and returns unique candidate points.

This is not yet an exact covering-radius or continuous Voronoi-cell theorem.
It is the first replayable candidate-generation receipt.  Exact covering,
dual-lattice, Voronoi, and Smith-normal-form refinements can be layered on this
surface without changing the diagnostic contract.

## Minimal Receipt

```python
import numpy as np

from LatticeForge.design_diagnostics import candidate_addition_diagnostics
from LatticeForge.refinement_candidates import refinement_candidate_points

base = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
])
candidates = refinement_candidate_points(np.eye(2), 2 * np.eye(2, dtype=int))
report = candidate_addition_diagnostics(base, candidates, grid_size=3)

print(report["candidates"][0]["candidate"])
```

The expected first candidate under approximate fill distance on the `3 x 3`
grid is the center point `(0.5, 0.5)`.
