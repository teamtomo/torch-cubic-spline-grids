from typing import Callable, Optional, Tuple

import einops
import torch
from typing_extensions import Self

from torch_cubic_spline_grids.utils import (
    MonotonicityType,
    batch,
    coerce_to_multichannel_grid,
)


class CubicSplineGrid(torch.nn.Module):
    """Base class for continuous parametrisations of multidimensional spaces."""

    ndim: int
    _data: torch.nn.Parameter
    _interpolation_function: Callable
    _interpolation_matrix: torch.Tensor
    _minibatch_size: int

    def __init__(
        self,
        resolution: Optional[Tuple[int, ...]] = None,
        n_channels: int = 1,
        minibatch_size: int = 1_000_000,
        monotonicity: Optional[MonotonicityType] = None,
    ):
        super().__init__()
        if resolution is None:
            resolution = (2,) * self.ndim
        grid_shape = (n_channels, *resolution)
        self.data = torch.zeros(size=grid_shape)
        self._minibatch_size = minibatch_size
        self._monotonicity = monotonicity
        self.register_buffer(
            name='interpolation_matrix',
            tensor=self._interpolation_matrix,
            persistent=False,
        )

    def _interpolate(self, u: torch.Tensor) -> torch.Tensor:
        return self._interpolation_function(
            self._data,
            u,
            matrix=self.interpolation_matrix,
            monotonicity=self._monotonicity,
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        u = self._coerce_to_batched_coordinates(u)  # (b, d)

        interpolated = [
            self._interpolate(minibatch_u)
            for minibatch_u in batch(u, n=self._minibatch_size)
        ]  # List[Tensor[(b, d)]]
        interpolated = torch.cat(interpolated, dim=0)  # (b, d)
        return self._unpack_interpolated_output(interpolated)

    @classmethod
    def from_grid_data(cls, data: torch.Tensor) -> Self:
        """Instantiate a grid from existing grid data.

        Parameters
        ----------
        data: torch.Tensor
            (c, *grid_dimensions) or (*grid_dimensions) array of multichannel values at
            each grid point.
        """
        grid = cls()
        grid.data = data
        return grid

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, grid_data: torch.Tensor) -> None:
        grid_data = coerce_to_multichannel_grid(grid_data, grid_ndim=self.ndim)
        self._data = torch.nn.Parameter(grid_data)

    @property
    def n_channels(self) -> int:
        return int(self._data.size(0))

    @property
    def resolution(self) -> Tuple[int, ...]:
        return tuple(self._data.shape[1:])

    def _coerce_to_batched_coordinates(self, u: torch.Tensor) -> torch.Tensor:
        u = torch.atleast_1d(torch.as_tensor(u, dtype=torch.float32))
        self._input_is_coordinate_like = u.shape[-1] == self.ndim
        if self._input_is_coordinate_like is False and self.ndim == 1:
            u = einops.rearrange(u, '... -> ... 1')  # add singleton coord dimension
        else:
            u = torch.atleast_2d(u)  # add batch dimension if missing
        u, self._packed_shapes = einops.pack([u], pattern='* coords')
        if u.shape[-1] != self.ndim:
            ndim = u.shape[-1]
            raise ValueError(
                f'Cannot interpolate on a {self.ndim}D grid with {ndim}D coordinates'
            )
        return u

    def _unpack_interpolated_output(self, interpolated: torch.Tensor) -> torch.Tensor:
        [interpolated] = einops.unpack(
            interpolated, packed_shapes=self._packed_shapes, pattern='* coords'
        )
        if self._input_is_coordinate_like is False and self.ndim == 1:
            interpolated = einops.rearrange(interpolated, '... 1 -> ...')
        return interpolated
