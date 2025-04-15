"""Cubic B-spline interpolation on multidimensional grids in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('torch-cubic-b-spline-grid')
except PackageNotFoundError:
    __version__ = 'uninstalled'

__author__ = 'Alister Burt'
__email__ = 'alisterburt@gmail.com'

from .b_spline_grids import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
)
from .catmull_rom_grids import (
    CubicCatmullRomGrid1d,
    CubicCatmullRomGrid2d,
    CubicCatmullRomGrid3d,
    CubicCatmullRomGrid4d,
)

__all__ = [
    'CubicBSplineGrid1d',
    'CubicBSplineGrid2d',
    'CubicBSplineGrid3d',
    'CubicBSplineGrid4d',
    'CubicCatmullRomGrid1d',
    'CubicCatmullRomGrid2d',
    'CubicCatmullRomGrid3d',
    'CubicCatmullRomGrid4d',
]
