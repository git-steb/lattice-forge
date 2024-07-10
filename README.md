# LatticeForge

LatticeForge is a Python library that allows you to create lattice bases, generate lattice point sets, compute Voronoi polytopes, and optimize designs for statistical models.

## Features

- Create lattice bases
- Generate lattice point sets
- Compute Voronoi polytopes
- Optimize designs for statistical models

## Installation

You can install the package using pip:

```
pip install LatticeForge
```


## Usage

Here's an example of how to use the library:

```python
import numpy as np
from LatticeForge import closestIndex

R = np.array([[1, 0], [0, 1]])
x = np.array([0.5, 0.5])
uhat = closestIndex(R, x)
print(uhat)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

