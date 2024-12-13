# LatticeForge

**LatticeForge** is a Python library for working with lattice-based designs, including creating lattice bases, generating lattice point sets, computing Voronoi polytopes, and optimizing designs for statistical models.

## Features

- **Create Lattice Bases**: Easily define and manipulate lattice bases.
- **Generate Lattice Point Sets**: Generate points within unit hypercubes using advanced rasterization techniques.
- **Handle Offsets**: Generate lattice points with optional non-zero offsets.
- **Compute Voronoi Polytopes**: Calculate Voronoi cells for lattice points.
- **Optimize Designs**: Useful for statistical models and sampling applications.

## Installation

You can install the package using `pip` from the root directory of the repository:

```bash
pip install -e .
```

### Dependencies

Make sure you have the following dependencies installed:

- **NumPy**
- **SciPy**
- **Eigen** (for compiled C++ extensions)

If building the C++ extensions fails, ensure your system has a compatible compiler installed (e.g., `gcc` or `clang`).

## Usage

### Example: Finding the Closest Lattice Point

Here's a basic example of using `closestIndex` to find the closest lattice point:

```python
import numpy as np
from LatticeForge import closestIndex

# Define a lattice basis
R = np.array([[1, 0], [0, 1]])

# Define a point
x = np.array([0.5, 0.5])

# Find the closest lattice point
uhat = closestIndex(R, x)
print(uhat)
```

### Example: Rasterizing a Lattice with an Offset

Generate lattice points within a unit hypercube with a non-zero offset:

```python
from LatticeForge import rasterize
import numpy as np

# Define a lattice basis
R = np.array([[1.02, 0.09], [0.01, 1.00]])

# Define an offset
xofs = np.array([0.5, 0.5])

# Rasterize lattice points with offset
points = rasterize(R, xofs=xofs, eps=1e-6, minbvol=True, sortit=True)
print(points)
```

## Testing

To run the test suite, use the following command:

```bash
python -m unittest discover tests
```

### Run a Specific Test

For example, to test rasterization with a non-zero offset:

```bash
python -m unittest tests.test_lattice_design_pipeline.TestLatticePipeline.test_rasterize_with_nonzero_xofs
```

## License

This project is licensed under the MIT License – see the `LICENSE` file for details.
