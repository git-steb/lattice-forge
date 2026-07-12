# Design Diagnostics

Date: 2026-07-12
Status: executable public surface

LatticeForge now exposes a small diagnostic layer for finite point sets. The
layer is intentionally modest: it does not decide that a design is optimal. It
gives stable quantities that can be recomputed while lattice refinements,
projection requirements, and Gaussian-process metric frames are being discussed.

## Quantities

For a finite design `X = {x_i}` in a bounded region:

- nearest-pair distance is `min_{i != j} ||x_i - x_j||`;
- separation is half the nearest-pair distance;
- fill distance is approximated on a deterministic tensor grid as the largest
  distance from a grid witness point to its nearest design point;
- mesh ratio is `fill_distance / separation`;
- projection diagnostics repeat the same reads on coordinate projections.

The current fill distance is a deterministic approximation, not a final
continuous optimization result. That is useful for paper work because it makes
small examples replayable and exposes which criterion changed after adding or
projecting points.

## Candidate Additions

The executable surface also supports before/after reads for new points:

- `addition_diagnostics(X, A, ...)` compares a design `X` with `X union A`;
- `candidate_addition_diagnostics(X, C, rank_by=...)` evaluates one candidate at
  a time and ranks candidates under the stated criterion;
- `diagnostic_delta(before, after)` records which scalar reads changed.

The supported ranking criteria are deliberately explicit:

```text
fill_distance                 minimize
mesh_ratio                    minimize
max_projected_fill_distance   minimize
max_projected_mesh_ratio      minimize
separation                    maximize
nearest_pair_distance         maximize
min_projected_separation      maximize
```

This answers a narrow but important sequential-design question: under a fixed
admissible candidate set, which point currently improves the chosen diagnostic?
It does not claim that the candidate set itself is globally optimal.

## GP Metric Frames

Gaussian-process length scales change the geometry in which distances should be
read. For diagonal length scales `theta`, LatticeForge evaluates distances after
the transform

```text
z_j = x_j / theta_j.
```

For a full positive definite metric `M`, distances are read as

```text
d_M(x, y)^2 = (x - y)^T M (x - y).
```

The public helper `metric_gram(G, ...)` computes

```text
Gamma_M = G.T @ M @ G
```

for a lattice basis `G`. This is a useful first invariant for reducing redundant
search: if two presentations have the same metric Gram matrix, they deserve to
be compared as possible equivalent metric-frame constructions before spending
more search effort on point generation or simulator evaluations.

## Use In The Paper Lane

The near-term statistical question is not simply "which point set is best?"
It is:

1. Which refinement moves are admissible under the lattice construction?
2. Which diagnostic changed after a point or batch was added?
3. Which of those changes still matter after the GP metric frame is applied?
4. Which projection requirements are preserved, improved, or broken?

This diagnostic layer answers the second, third, and fourth questions for small
finite examples now. The first question still belongs to the construction layer:
the candidate set should be supplied by lattice refinement, integer dilation,
or another clearly stated admissibility rule. The later typed Haskell reference
core can then own the exact construction and certificates, while this public
Python surface remains a readable statistical inspection layer.
