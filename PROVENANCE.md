# Provenance

This record identifies the implementation lineage of the public LatticeForge
surface. It is an engineering provenance record, not a determination of
inventorship, copyright ownership or patent rights.

## Project lineage

LatticeForge is the maintained Python/C++ companion to Steven Bergner's earlier
RotGrid MATLAB/Octave implementation:

<https://github.com/git-steb/rotgrid-matlab>

The legacy repository records the construction and use of rotational lattice
designs from 2008 onward. In particular, `lattice/samplecube.m` and
`lattice/rasterize.m` developed bounded-cube enumeration, dimension-incremental
address extension, conservative pruning and memory-aware traversal between
2012 and 2017.

`LatticeForge/bounded_sampling.py` is a modern, tested recovery of that public
enumeration principle. Its address-preserving iterator changes how results may
be consumed; it does not define a new lattice, rotation, refinement path or
statistical selection rule.

`LatticeForge/refinement_candidates.py` provides the repository's current
quotient/coset candidate surface. The design-diagnostic modules provide
explicit geometric and metric-frame reads of already constructed point sets.

## Authorship evidence

The Git history currently attributes the LatticeForge implementation outside
the vendored Eigen tree to Steven Bergner. Commit metadata is supporting
evidence, not a substitute for an ownership or inventorship review.

The legacy MATLAB repository separately identifies a `thierry/` directory as
code authored by Thierry Blu and an `others/` directory as code from other
authors. Those directories are not the source of the recovered bounded sampler
described above. Any future translation from those directories requires an
explicit source-level attribution and licence review before admission here.

## Third-party source

The vendored Eigen headers retain their upstream file notices and licence
bundle. See `THIRD_PARTY_NOTICES.md` and `LatticeForge/eigen/COPYING.README`.

## Publication boundary

The intended Paper 1 companion contains only construction, bounded
enumeration, declared diagnostics and reproducibility support for material
already disclosed in the paper or its published lineage. Stronger unpublished
theorems, adaptive selection research and private execution mechanisms are not
part of this repository's release surface.
