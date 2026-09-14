# Paper 1 public release boundary

Date: 2026-09-13

Status: release-candidate boundary; no publication is implied by this file.

## Purpose

LatticeForge is the smallest executable companion needed to inspect and
reproduce the construction layer of *Low-rate rotational lattice refinement
for sequential computer experiments*. The repository is not the complete
research implementation behind later work.

The operating distinction is:

> The lattice construction determines what is lawfully addable. A declared
> statistical criterion determines what is useful for a particular analysis.
> A record makes that decision reproducible.

The public package may construct and expose lawful candidates and may compute
declared diagnostics. It does not treat a diagnostic ranking as a theorem or a
general performance guarantee.

## Included surface

- lattice bases and integer dilations;
- bounded-region enumeration with aligned integer addresses;
- memory-bounded and pull-based traversal of the same finite design;
- quotient/coset representatives and one-step refinement candidates;
- geometric and GP-metric-frame diagnostics;
- deterministic examples, tests and provenance records.

## Held outside this release

- unpublished theorem search or generalized construction claims;
- private adaptive-selection, sensitivity or continuation research beyond the
  manuscript's stated example;
- private solver, preconditioner or execution mechanisms;
- internal research-framework architecture;
- application-specific confidential data or models.

## Language boundary

Python is the present inspection and package surface. C++ owns one narrow
nearest-point kernel. R remains the natural statistical interface, while a
typed Haskell implementation may serve as an independent construction
reference.

Rust is a possible future deployment kernel, not an accepted rewrite. A Rust
implementation must first reproduce the same bounded pages and refusal cases,
then demonstrate a measured packaging, memory or browser-deployment advantage.
If admitted, Python, R and browser bindings should remain thin consumers of one
versioned record contract.

## Release gate

A public release is ready only when:

1. every exposed operation is already supported by Paper 1 or the published
   implementation lineage;
2. provenance and third-party notices are complete;
3. the package metadata and distributed licence files agree;
4. tests pass from a clean source distribution;
5. the release tag identifies the exact manuscript/code pairing.
