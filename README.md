# LatticeForge

LatticeForge is a research prototype for constructing and inspecting
lattice-based designs for computer experiments. It focuses on integer lattice
bases, dilation matrices, nested point sets, Voronoi geometry, and
space-filling diagnostics for Gaussian-process emulation and related statistical
design problems.

It is being prepared as the bounded software companion to the manuscript
*Low-rate rotational lattice refinement for sequential computer experiments*,
which is under preparation for submission; this repository does not claim the
larger unpublished research programme.

The current code is intentionally close to the design-construction layer: it
generates lattice point sets in bounded regions, studies refinement by integer
dilation, and exposes small visual and testable surfaces for checking how a
candidate design occupies space.

## Current Surface

- Lattice basis and point-set construction in `LatticeForge/`
- Memory-bounded unit-cube rasterization in `LatticeForge/bounded_sampling.py`,
  with both a conventional NumPy result and an address-preserving batch
  iterator
- Nearest-lattice-point utilities, with a compiled C++ extension path where
  useful
- Search and verification helpers for integer dilation matrices
- Coset-derived one-step refinement candidates from an integer dilation matrix,
  with quotient representative and offset provenance retained per batch
- Deterministic design diagnostics for separation, approximate fill distance,
  mesh ratio, projection behavior, and GP metric-frame reads
- Candidate-addition diagnostics that rank one-point continuations by explicit
  criteria such as fill distance, mesh ratio, separation, or projected reads
- Voronoi-based visualization of nested lattice designs in
  `notebooks/voronoi_lattice_visualization.py`
- A static notebook rendering under GitHub Pages:
  <https://git-steb.github.io/lattice-forge/>
- Implementation steering notes:
  `docs/implementation-direction.md` and
  `docs/haskell-reference-core.md`
- Diagnostic note:
  `docs/design-diagnostics.md`
- Refinement-candidate note:
  `docs/refinement-candidates.md`

## Why This Repository Exists

Sequential computer experiments often need designs that remain useful after
each additional batch. Lattice constructions give a concrete way to study that
problem: refinement, spacing, projection behavior, and local neighborhoods can
be made explicit enough to inspect, test, and eventually optimize.

This repository is the companion code surface for that design work. The current
interface is research-facing and may change as the design-construction layer is
tightened.

The construction determines which refinements are admissible. A statistical
criterion may rank those candidates for a particular analysis, but that ranking
does not become a construction theorem or a universal performance claim. The
full inclusion and exclusion boundary is recorded in
[`docs/public-release-boundary.md`](docs/public-release-boundary.md).

## Install

Use Python 3.10 or later. From the repository root:

```bash
pip install -e .
```

The package expects NumPy, SciPy, matplotlib, SymPy, pybind11, and a compiler
toolchain for the C++ extension.

For an isolated local check with `uv`:

```bash
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python -e .
```

## Quick Checks

Run the standard test discovery command:

```bash
.venv/bin/python -m unittest discover tests
```

Check the Voronoi visualization script directly:

```bash
.venv/bin/python -m py_compile notebooks/voronoi_lattice_visualization.py
```

## Example

```python
import numpy as np
from LatticeForge.refinement_candidates import refinement_candidate_points
from LatticeForge.design_diagnostics import design_diagnostics

G = np.eye(2)
K = 2 * np.eye(2, dtype=int)

candidates = refinement_candidate_points(G, K)
print(candidates)

report = design_diagnostics(candidates, grid_size=11, length_scales=np.array([0.8, 1.2]))
print(report["separation"], report["fill_distance"], report["mesh_ratio"])
```

For a larger bounded design, consume construction batches without first
forming the complete point matrix:

```python
from LatticeForge import CubeTraversalDiagnostics, iter_unit_cube_batches

diagnostics = CubeTraversalDiagnostics()
for batch in iter_unit_cube_batches(
    G,
    shift=np.zeros(G.shape[0]),
    max_memory_mib=32,
    diagnostics=diagnostics,
):
    addresses = batch.addresses
    points = batch.points
    # Store, plot, or evaluate this bounded batch.

print(diagnostics.to_dict())
```

This is the bounded-region operation in the RotGrid construction: determine a
finite integer-address enclosure, generate physical lattice points, and keep
those inside the declared half-open cube. The iterator changes memory use, not
the design. It does not select a lattice, rotation, shift, refinement path, or
statistical criterion.

## Status

Active research code. Interfaces may change while the design-construction
story is being tightened.

Before preparing a release, run:

```bash
python tools/check_release.py
python -m unittest discover tests
```

## Licence, provenance and citation

Original LatticeForge code and documentation are available under the
MIT licence in [`LICENSE`](LICENSE). Vendored Eigen headers retain
their own terms; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The implementation lineage and the boundary around earlier collaborator code
are recorded in [`PROVENANCE.md`](PROVENANCE.md). Citation metadata is provided
in [`CITATION.cff`](CITATION.cff).
