# Haskell Reference Core

Date: 2026-07-12
Status: near-term implementation plan

LatticeForge should grow a typed Haskell reference core while preserving Python
and R as the first statistical-facing surfaces.

The point of the Haskell layer is not to make every user install GHC. The point
is to give the mathematical construction a precise executable owner: lattice
bases, dilation matrices, bounded regions, refinements, projections, criteria,
and certificates should be represented by types and checked by small replayable
receipts before they are exposed through notebooks or statistical package
interfaces.

## Core Boundary

The Haskell core owns exact construction and verification:

- formation of lattice bases and bounded design regions;
- admissible integer dilations and refinement indices;
- nested point-set generation with explicit offsets and region boundaries;
- projection diagnostics as algebraic queries over a design;
- certificates that a candidate refinement satisfies the stated construction
  and diagnostic checks.

The statistical surface owns interpretation:

- Gaussian-process length scales and correlation geometry;
- design criteria such as fill distance, separation, discrepancy, and
  projection quality;
- comparison against Latin-hypercube, maximin, or other experimental-design
  baselines;
- user-facing visualization, notebooks, and paper figures.

This separation is deliberate. A lattice construction determines what can be
added lawfully. A statistical model determines what currently appears valuable
to add.

## First Haskell Objects

The first reference-core pass should be small:

```haskell
data Basis
data Dilation
data Region
data Offset
data Design
data Refinement
data Projection
data Criterion
data Certificate
```

Those names are placeholders until implementation fixes their exact shapes.
Their purpose is to keep the first pass focused on the design-construction
contract rather than on a complete statistical runtime.

## First Receipts

The first useful receipts should be executable checks, not broad claims:

1. Reproduce small two-dimensional examples from the current Python package.
2. Verify determinant/refinement counts for simple integer dilation matrices.
3. Confirm that nested designs preserve previously admitted points.
4. Compute projection diagnostics from the same design in Haskell and Python.
5. Export a small neutral data table that R and Python can both read.

Only after those receipts exist should any user-facing API depend on the
Haskell implementation.

## Public Interface Posture

Near term:

- Python remains the inspection and notebook surface.
- R is the intended statistical package surface.
- C++ remains a narrow accelerated kernel lane where it is already useful.
- Haskell grows as a reference core and possible kernel generator.
- Julia and Rust remain later bridge or packaging candidates, not current
  owners of the design algebra.

The practical target is a repository that a statistician can inspect now, while
the exact algebra becomes strong enough to support future R, Python, or compiled
interfaces without changing the mathematical story each time the surface
language changes.

## Connection to the Paper Lane

The current paper problem is sequential design under refinement: how to add
points without losing useful structure already obtained. The Haskell reference
core should therefore serve the paper by making three things explicit:

- the admissible refinement moves;
- the projection and spacing diagnostics affected by those moves;
- the evidence that a proposed construction can be replayed independently of a
  particular notebook state.

That is the first useful bridge between lattice construction, Gaussian-process
experimental design, and a typed implementation.
