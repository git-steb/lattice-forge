# Third-party notices

LatticeForge contains and depends on third-party software. The MIT
licence in `LICENSE` applies to the original LatticeForge code and
documentation; it does not replace the licences of third-party components.

## Eigen

The source distribution includes a vendored subset of Eigen under
`LatticeForge/eigen/`. Eigen is primarily licensed under the Mozilla Public
License 2.0, with individual files under other third-party licences (listed
below). The licence texts are preserved beside the vendored source:

- `LatticeForge/eigen/COPYING.README`
- `LatticeForge/eigen/COPYING.MPL2`
- `LatticeForge/eigen/COPYING.BSD`
- `LatticeForge/eigen/COPYING.MINPACK`
- `LatticeForge/eigen/COPYING.APACHE`
- `LatticeForge/eigen/COPYING.LGPL`

Upstream project: <https://gitlab.com/libeigen/eigen>

The version macro in the vendored `Eigen/src/Core/util/Macros.h` reads 3.2.10,
but the tree also contains files that postdate that release; the exact upstream
version or commit of the vendored tree has not been established.

### Files derived from LGPL-licensed code

Three vendored Eigen headers state in their own notices that they derive from
third-party code, and they are kept with those notices:

- `Eigen/src/OrderingMethods/Amd.h`, adapted from the CSparse library
  (T. A. Davis), licensed under the GNU Lesser General Public License,
  version 2.1 or later;
- `Eigen/src/SparseCholesky/SimplicialCholesky_impl.h`, adapted from the LDL
  library (T. A. Davis), under the same licence;
- `Eigen/src/IterativeLinearSolvers/IncompleteLUT.h`, derived from the ILUT
  code in SPARSKIT (Y. Saad), whose header records that the original author
  permitted relicensing it under the Mozilla Public License 2.0.

The text of the GNU LGPL 2.1 is in `LatticeForge/eigen/COPYING.LGPL`; it is the
text distributed with the R project. These files are not part of the compiled
LatticeForge extension: building it with pybind11 3.1 reaches only headers
under `Eigen/Core` and `Eigen/SparseCore`.

## Installed dependencies

NumPy, SciPy, matplotlib, SymPy and pybind11 are installed as dependencies and
are not relicensed by this repository. Their own distributions carry their
respective licence terms.
