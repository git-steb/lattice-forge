# Changelog

Versions mark planned steps, not every change. A minor version (0.x) is
reserved for a step on the project roadmap. A patch version (0.1.x) records a
correction that changes what existing callers receive.

## 0.1.1 (unreleased)

### Fixed

- `rasterize(R, xofs=...)` now returns the complete set of lattice points in
  the closed unit cube. Version 0.1.0 chose its integer grid without applying
  the shift `xofs`, so with a nonzero shift it could omit most of the correct
  points: in the cases checked against an independent enumeration, between 50%
  and 98% of them. Calls without a shift are unchanged.

### Changed

- `samplecube` and `rasterize` traverse the integer address box in
  memory-bounded chunks (`maxmemMB`). In the comparison with 0.1.0 on random
  lattices in dimensions 2 to 4, `samplecube` returned the same points in the
  same order.
- Python 3.10 or later is required (0.1.0 declared 3.6).

### Added

- Pull-based, address-preserving iteration over a bounded design:
  `iter_unit_cube_batches`, `sample_unit_cube`, `unit_cube_address_bounds`,
  `CubePointBatch` and `CubeTraversalDiagnostics`.
- A `LICENSE` file (MIT, the licence the package metadata has declared since
  0.1.0), third-party notices that include the LGPL-derived Eigen headers,
  provenance and citation metadata, and `tools/check_release.py`.

## 0.1.0

First public version.
