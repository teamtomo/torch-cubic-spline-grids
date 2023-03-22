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

`torch_cubic_spline_grids` provides a set of PyTorch 


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
