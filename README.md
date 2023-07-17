# torch-cubic-spline-grids

[![License](https://img.shields.io/pypi/l/torch-cubic-spline-grids.svg?color=green)](https://github.com/alisterburt/torch-cubic-spline-grids/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-cubic-spline-grids.svg?color=green)](https://pypi.org/project/torch-cubic-spline-grids)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-cubic-spline-grids.svg?color=green)](https://python.org)
[![CI](https://github.com/alisterburt/torch-cubic-spline-grids/actions/workflows/ci.yml/badge.svg)](https://github.com/alisterburt/torch-cubic-spline-grids/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/alisterburt/torch-cubic-spline-grids/branch/main/graph/badge.svg)](https://codecov.io/gh/alisterburt/torch-cubic-spline-grids)

*Cubic spline interpolation on multidimensional grids in PyTorch.*

The primary goal of this package is to provide learnable, continuous
parametrisations of 1-4D spaces.

--- 

## Overview

`torch_cubic_spline_grids` provides a set of PyTorch components called grids.

Grids are defined by
- their dimensionality (1d, 2d, 3d, 4d...)
- the number of points covering each dimension (`resolution`)
- the number of values stored on each grid point (`n_channels`)
- how we interpolate between values on grid points

All grids in this package consist of uniformly spaced points covering the full 
extent of each dimension.

### First steps
Let's make a simple 2D grid with one value on each grid point.

```python
import torch
from torch_cubic_spline_grids import CubicBSplineGrid2d

grid = CubicBSplineGrid2d(resolution=(5, 3), n_channels=1)
```

- `grid.ndim` is `2`
- `grid.resolution` is `(5, 3)` (or `(h, w)`) 
- `grid.n_channels` is `1`
- `grid.data.shape` is `(1, 5, 3)` (or `(c, h, w)`)

In words, the grid extends over two dimensions `(h, w)` with 5 points 
in `h` and `3` points in `w`. 
There is one value stored at each point on the 2D grid. 
The grid data is stored in a tensor of shape `(c, *grid_resolution)`.

We can obtain the value (interpolant) at any continuous point on the grid.
The grid coordinate system extends from `[0, 1]` along each grid dimension. 
The interpolant is obtained by sequential application of
cubic spline interpolation along each dimension of the grid.

```python
coords = torch.rand(size=(10, 2))  # values in [0, 1]
interpolants = grid(coords)
```

- `interpolants.shape` is `(10, 1)`

### Optimisation

Values at each grid point can be optimised by minimising a loss function associated with grid interpolants. 
In this way the continuous space of the grid can be made to more accurately model a 1-4D space.

<p align="center" width="100%">
    <img width="60%" src="https://user-images.githubusercontent.com/7307488/226992179-049a63a0-a2f3-4432-b38e-6e8bcaa6a4a8.png">
</p>

The image above shows the values of 6 control points on a 1D grid being optimised such 
that interpolating between them with cubic B-spline interpolation approximates a single oscillation of a sine wave. 

Notebooks are available for this 
[1D example](./examples/optimise_1d_grid_model.ipynb) 
and a similar 
[2D example](./examples/optimise_2d_grid_model.ipynb).

### Types of grids

`torch_cubic_spline_grids` provides grids which can be interpolated with **cubic 
B-spline** interpolation or **cubic Catmull-Rom spline** interpolation. 

| spline             | continuity | interpolating? |
|--------------------|------------|----------------|
| cubic B-spline     | C2         | No             |
| Catmull-Rom spline | C1         | Yes            |

If your need the resulting curve to intersect the data on the grid you should
use the cubic Catmull-Rom spline grids 

- `CubicCatmullRomGrid1d`
- `CubicCatmullRomGrid2d`
- `CubicCatmullRomGrid3d`
- `CubicCatmullRomGrid4d`

If you require continuous second derivatives then the cubic B-spline grids are more 
suitable.

- `CubicBSplineGrid1d`
- `CubicBSplineGrid2d`
- `CubicBSplineGrid3d`
- `CubicBSplineGrid4d`

### Regularisation

The number of points in each dimension should be chosen such that interpolating on the 
grid can approximate the underlying phenomenon being modelled without overfitting. 
A low resolution grid provides a regularising effect by smoothing the model.


## Installation

`torch_cubic_spline_grids` is available on PyPI

```shell
pip install torch-cubic-spline-grids
```


## Related work

This is a PyTorch implementation of the way
[Warp](http://warpem.com/warp/#) models continuous deformation
fields and locally variable optical parameters in cryo-EM images. 
The approach is described in
[Dimitry Tegunov's paper](https://doi.org/10.1038/s41592-019-0580-y):

> Many methods in Warp are based on a continuous parametrization of 1- to
> 3-dimensional spaces.
> This parameterization is achieved by spline interpolation between points on a coarse,
> uniform grid, which is computationally efficient.
> A grid extends over the entirety of each dimension that needs to be modeled.
> The grid resolution is defined by the number of control points in each dimension
> and is scaled according to physical constraints
> (for example, the number of frames or pixels) and available signal.
> The latter provides regularization to prevent overfitting of sparse data with too many
> parameters.
> When a parameter described by the grid is retrieved for a point in space (and time),
> for example for a particle (frame), B-spline interpolation is performed at that point
> on the grid.
> To fit a grid’s parameters, in general, a cost function associated with the
> interpolants at specific positions on the grid is optimized. 

---

For a fantastic introduction to splines I recommend 
[Freya Holmer](https://www.youtube.com/watch?v=jvPPXbo87ds)'s YouTube video.

[The Continuity of Splines - YouTube](https://youtu.be/jvPPXbo87ds)
