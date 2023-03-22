from functools import partial
from typing import Tuple, Callable, Union, Sequence, Optional

import torch

from torch_cubic_spline_grids._base_cubic_grid import CubicSplineGrid
from torch_cubic_spline_grids.interpolate_grids import (
    interpolate_grid_1d as _interpolate_grid_1d,
    interpolate_grid_2d as _interpolate_grid_2d,
    interpolate_grid_3d as _interpolate_grid_3d,
    interpolate_grid_4d as _interpolate_grid_4d,
)
from torch_cubic_spline_grids._constants import CUBIC_B_SPLINE_MATRIX

CoordinateLike = Union[float, Sequence[float], torch.Tensor]


class CubicBSplineGrid1d(CubicSplineGrid):
    """Continuous parametrisation of a 1D space with a specific resolution."""
    ndim: int = 1
    _interpolation_function: Callable = partial(
        _interpolate_grid_1d, matrix=CUBIC_B_SPLINE_MATRIX
    )

    def __init__(
        self,
        resolution: Optional[Union[int, Tuple[int]]] = None,
        n_channels: int = 1
    ):
        if isinstance(resolution, int):
            resolution = tuple([resolution])
        super().__init__(resolution, n_channels)


class CubicBSplineGrid2d(CubicSplineGrid):
    """Continuous parametrisation of a 2D space with a specific resolution."""
    ndim: int = 2
    _interpolation_function: Callable = partial(
        _interpolate_grid_2d, matrix=CUBIC_B_SPLINE_MATRIX
    )


class CubicBSplineGrid3d(CubicSplineGrid):
    """Continuous parametrisation of a 3D space with a specific resolution."""
    ndim: int = 3
    _interpolation_function: Callable = partial(
        _interpolate_grid_3d, matrix=CUBIC_B_SPLINE_MATRIX
    )


class CubicBSplineGrid4d(CubicSplineGrid):
    """Continuous parametrisation of a 4D space with a specific resolution."""
    ndim: int = 4
    _interpolation_function: Callable = partial(
        _interpolate_grid_4d, matrix=CUBIC_B_SPLINE_MATRIX
    )
