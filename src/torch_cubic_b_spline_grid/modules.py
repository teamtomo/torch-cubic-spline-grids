import torch
import einops

from torch_cubic_b_spline_grid.interpolate_grid import (
    interpolate_grid_1d as _interpolate_grid_1d,
    interpolate_grid_2d as _interpolate_grid_2d,
    interpolate_grid_3d as _interpolate_grid_3d,
    interpolate_grid_4d as _interpolate_grid_4d,
)


class CubicBSplineGrid1d(torch.nn.Module):
    """Continuous parametrisation of a 1D space with a specific resolution."""
    def __init__(self, resolution: int = 1, ndim: int = 1):
        super().__init__()
        self._data = torch.nn.Parameter(torch.zeros((resolution, ndim)))

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, tensor: torch.Tensor):
        if tensor.ndim == 1:
            tensor = einops.rearrange(tensor, 'w -> w 1')
        self._data = torch.nn.Parameter(tensor)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _interpolate_grid_1d(grid=self._data, u=u)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        grid = cls()
        grid.data = tensor
        return grid


class CubicBSplineGrid2d(torch.nn.Module):
    def __init__(self, resolution: tuple[int, int] = (1, 1), ndim: int = 1):
        super().__init__()
        h, w = resolution
        self._data = torch.nn.Parameter(torch.zeros((h, w, ndim)))

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, tensor: torch.Tensor):
        if tensor.ndim == 2:
            tensor = einops.rearrange(tensor, 'h w -> h w 1')
        self._data = torch.nn.Parameter(tensor)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _interpolate_grid_2d(grid=self._data, u=u)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        grid = cls()
        grid.data = tensor
        return grid


class CubicBSplineGrid3d(torch.nn.Module):
    def __init__(self, resolution: tuple[int, int, int] = (1, 1, 1), ndim: int = 1):
        super().__init__()
        d, h, w = resolution
        self._data = torch.nn.Parameter(torch.zeros((d, h, w, ndim)))

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, tensor: torch.Tensor):
        if tensor.ndim == 3:
            tensor = einops.rearrange(tensor, 'd h w -> d h w 1')
        self._data = torch.nn.Parameter(tensor)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _interpolate_grid_3d(grid=self._data, u=u)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        grid = cls()
        grid.data = tensor
        return grid


class CubicBSplineGrid4d(torch.nn.Module):
    def __init__(
        self, resolution: tuple[int, int, int, int] = (1, 1, 1, 1), ndim: int = 1
    ):
        super().__init__()
        t, d, h, w = resolution
        self._data = torch.nn.Parameter(torch.zeros((t, d, h, w, ndim)))

    @property
    def data(self) -> torch.Tensor:
        return self._data.detach()

    @data.setter
    def data(self, tensor: torch.Tensor):
        if tensor.ndim == 4:
            tensor = einops.rearrange(tensor, 't d h w -> t d h w 1')
        self._data = torch.nn.Parameter(tensor)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return _interpolate_grid_4d(grid=self._data, u=u)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor):
        grid = cls()
        grid.data = tensor
        return grid


if __name__ == '__main__':
    grid = CubicBSplineGrid1d(resolution=5, ndim=1)
    print(list(grid.parameters()))
    grid = CubicBSplineGrid1d.from_tensor(torch.zeros(5))
    print(list(grid.parameters()))