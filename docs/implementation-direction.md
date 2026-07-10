# Implementation Direction

Date: 2026-07-10
Status: steering note

This repository should stay clear about what is strong now and what should
change next. LatticeForge is strongest when it remains close to statistical
design construction: lattice bases, integer dilations, nested refinements,
bounded regions, nearest-lattice-point queries, Voronoi neighborhoods,
projection behavior, and space-filling diagnostics.

The implementation should preserve that strength while avoiding premature
commitment to a single language ecosystem.

## Current Posture

The present code base is a useful research-facing prototype:

- Python gives a familiar notebook and package surface.
- NumPy and SciPy make quick statistical and geometric experiments available.
- C++/Eigen provides one narrow accelerated kernel for nearest-lattice-point
  work.
- CMake currently mixes native compilation and notebook rendering.

That shape is acceptable for exploration, but it should not become the final
architecture by accident.

## Dependency Rule

Free software is not automatically a light dependency.

GHC, Haskell, C++, Rust, Julia, Python, and R are all viable in different ways.
For this project, dependency weight is judged by the user and contributor path:

- Can a statistician install it without learning a new compiler stack?
- Can an R user call it from ordinary package workflows?
- Can a Python user inspect examples without building the full research core?
- Can the mathematical core remain typed, testable, and independent of the
  surface language?

For now, Haskell should not become a hard install dependency of the public
Python package. It should enter first as a typed reference core and as a source
of generated or wrapped kernels once the boundary is clear.

## Language Roles

### Python

Python remains the thin user and demonstration layer. It is good for notebooks,
plots, examples, exploratory scripts, and comparison against NumPy/SciPy
baselines. New mathematical ownership should not be placed in Python unless the
code is deliberately temporary or demonstrational.

### C++

C++ remains acceptable for narrow, performance-critical kernels and stable C ABI
surfaces. It is already natural for R package users through established compiled
code workflows, and it sits close to the compiler/tooling world used by many
numerical libraries.

C++ should not become the conceptual owner of the design algebra. Use it where a
kernel has been isolated, tested, and shown to need low-level performance.

### Haskell

Haskell is the preferred candidate for the typed mathematical core. The first
Haskell layer should model the design objects directly:

- `Basis`
- `Dilation`
- `Region`
- `Design`
- `Refinement`
- `Projection`
- `Criterion`
- `Certificate`

The Haskell core should begin as reference implementation plus executable
checks. It should be compared against the existing Python/C++ behavior before it
becomes a dependency of user-facing packages.

### R

R is the primary statistical-facing ecosystem. The near-term aim is not to
replace R, but to make LatticeForge useful from R through ordinary package
interfaces. The likely path is a small R package or bridge that calls stable
compiled entry points and presents results as familiar R vectors, matrices,
data frames, and possibly Arrow-backed tables.

R compatibility should shape the public boundary even when the internal core is
not written in R.

### Julia

Julia is not bypassed permanently, but it is not a near-term architectural
owner. It is strong for numerical computing and could become a useful comparison
or wrapper target. Adding Julia now would add another runtime and package
surface before the design algebra has been stabilized.

Treat Julia as a later interface or benchmark lane, not the next core rewrite.

### Rust

Rust is credible for safe native package development and R integration. It may
become useful for packaging, memory safety, or stable low-level kernels. It
should not displace Haskell as the typed design-language candidate unless a
specific packaging or performance boundary demands it.

## Near-Term Plan

Over the next few weeks, keep the work small and defensible:

1. Preserve the current Python package and examples as the public inspection
   surface.
2. Isolate the mathematical objects currently implicit in Python:
   basis, dilation, region, refinement, projection, and criterion.
3. Add golden tests that record current behavior for small dimensions and simple
   determinant/refinement cases.
4. Start a Haskell reference core outside the default Python install path.
5. Compare Haskell outputs against the Python/C++ fixtures before moving any
   public API.
6. Keep the C++ nearest-index kernel as benchmark/reference until a clearer
   replacement exists.
7. Sketch the R-facing boundary only after the typed core has at least one
   replayable design-construction receipt.

## Non-Goals

- Do not make GHC a required dependency for ordinary Python installation yet.
- Do not move algorithm ownership into notebooks.
- Do not add Julia or Rust only because they are attractive ecosystems.
- Do not promise a broad statistical language runtime.
- Do not expand the public surface faster than the testable design algebra.

## External Interface Anchors

- R packages already support compiled code through established source package
  installation workflows.
- GHC can build shared libraries and expose C-callable APIs, but doing so
  requires deliberate runtime and deployment handling.
- Arrow is a strong candidate for later table/vector interchange, especially
  where R and Python should share data without forcing one language to own the
  entire workflow.
- Julia has a C embedding API and can call C/Fortran directly, which makes it a
  plausible later bridge, but not a reason to add a Julia layer now.
- Rust has active R-package tooling through extendr/rextendr, which makes it a
  credible future native packaging option.
