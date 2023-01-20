"""Cubic B-spline interpolation on multidimensional grids in PyTorch"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("torch-cubic-b-spline-grid")
except PackageNotFoundError:
    __version__ = "uninstalled"

__author__ = "Alister Burt"
__email__ = "alisterburt@gmail.com"

from .modules import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
)