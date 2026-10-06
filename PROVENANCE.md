# Provenance

This record identifies the implementation lineage of the public LatticeForge
surface. It is an engineering provenance record, not a determination of
inventorship, copyright ownership or patent rights.

## Project lineage

The published lineage of the rotational lattice construction is:

- S. Bergner, D. Van De Ville, T. Blu and T. Möller, "On sampling lattices
  with similarity scaling relationships", SampTA 2009,
  <https://bigwww.epfl.ch/publications/bergner0901.html>;
- S. Bergner, *Making Choices in Multi-Dimensional Parameter Spaces*, PhD
  thesis, Simon Fraser University, 2011, Chapter 3,
  <https://summit.sfu.ca/item/12089>;
- K. Heitmann et al., "The Mira–Titan Universe: Precision Predictions for Dark
  Energy Surveys", Astrophys. J. 820 (2016) 108, Section 3, which used the
  nested lattice design for an emulator campaign.

LatticeForge is the maintained Python/C++ companion to Steven Bergner's earlier
RotGrid MATLAB/Octave implementation. That legacy implementation is not
publicly released; it is cited here as engineering provenance only. From 2012
to 2017 its bounded-cube enumeration developed dimension-incremental address
extension, conservative pruning and memory-aware traversal.

`LatticeForge/bounded_sampling.py` is a modern, tested reimplementation of that
enumeration principle, which is standard bounded lattice-point enumeration
rather than a new result. Its address-preserving iterator changes how results may
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
