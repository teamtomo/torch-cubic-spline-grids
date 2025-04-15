from typing import Optional

import einops
import torch

from torch_cubic_spline_grids.interpolate_pieces import (
    interpolate_pieces_1d,
    interpolate_pieces_2d,
    interpolate_pieces_3d,
    interpolate_pieces_4d,
)
from torch_cubic_spline_grids.pad_grids import (
    pad_grid_1d,
    pad_grid_2d,
    pad_grid_3d,
    pad_grid_4d,
)
from torch_cubic_spline_grids.utils import (
    MonotonicityType,
    interpolants_to_interpolation_data_1d,
    transform_to_monotonic_nd,
)


def interpolate_grid_1d(
    grid: torch.Tensor,
    u: torch.Tensor,
    matrix: torch.Tensor,
    monotonicity: Optional[MonotonicityType] = None,
) -> torch.Tensor:
    """Uniform cubic spline interpolation on a 1D grid.

    The range [0, 1] covers all data points in the 1D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(c, w)` array of `w` values in `c` channels to be interpolated.
    u: torch.Tensor
        `(b, 1)` array of query points in the range `[0, 1]` covering the `w`
        dimension of `grid`.
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.
    monotonicity: str
        when either 'increasing' or 'decreasing' is specified, ensures
        that control points of spline are monotonic.

    Returns
    -------
    interpolated: torch.Tensor
        `(b, c)` array of interpolated values in each channel.
    """
    if grid.ndim == 1:
        grid = einops.rearrange(grid, 'w -> 1 w')
    _, w = grid.shape

    # handle interpolation at edges by extending grid of control points according to
    # local gradients
    grid = pad_grid_1d(grid)

    # find control point indices and interpolation coordinate
    idx, t = interpolants_to_interpolation_data_1d(u[:, 0], n_samples=w)
    if monotonicity:
        grid = transform_to_monotonic_nd(grid, ndims=1, monotonicity=monotonicity)
    control_points = grid[..., idx]  # (c, b, 4)
    control_points = einops.rearrange(control_points, 'c b p -> b c p')

    # interpolate
    return interpolate_pieces_1d(control_points, t, matrix=matrix)


def interpolate_grid_2d(
    grid: torch.Tensor,
    u: torch.Tensor,
    matrix: torch.Tensor,
    monotonicity: Optional[MonotonicityType] = None,
) -> torch.Tensor:
    """Uniform cubic B-spline interpolation on a 2D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(c, h, w)` multichannel 2D grid.
    u: torch.Tensor
        `(b, 2)` array of values in the range `[0, 1]`.
        `[0, 1]` in `u[:, 0]` covers dim -2 (h) of `grid`
        `[0, 1]` in `u[:, 1]` covers dim -1 (w) of `grid`
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.
    monotonicity: str
        when either 'increasing' or 'decreasing' is specified, ensures
        that control points of spline are monotonic.

    Returns
    -------
    `(b, c)` array of interpolated values in each channel.
    """
    if grid.ndim == 2:
        grid = einops.rearrange(grid, 'h w -> 1 h w')
    _, h, w = grid.shape

    # pad grid to handle interpolation at edges.
    grid = pad_grid_2d(grid)

    # find control point indices and interpolation coordinate in each dim
    idx_h, t_h = interpolants_to_interpolation_data_1d(u[:, 0], n_samples=h)
    idx_w, t_w = interpolants_to_interpolation_data_1d(u[:, 1], n_samples=w)

    # construct (4, 4) grids of control points and 2D interpolant then interpolate
    idx_h = einops.repeat(idx_h, 'b h -> b h w', w=4)
    idx_w = einops.repeat(idx_w, 'b w -> b h w', h=4)
    if monotonicity:
        grid = transform_to_monotonic_nd(grid, ndims=2, monotonicity=monotonicity)
    control_points = grid[..., idx_h, idx_w]  # (c, b, 4, 4)
    control_points = einops.rearrange(control_points, 'c b h w -> b c h w')

    t = einops.rearrange([t_h, t_w], 'hw b -> b hw')
    return interpolate_pieces_2d(control_points, t, matrix=matrix)


def interpolate_grid_3d(
    grid: torch.Tensor,
    u: torch.Tensor,
    matrix: torch.Tensor,
    monotonicity: Optional[MonotonicityType] = None,
) -> torch.Tensor:
    """Uniform cubic B-spline interpolation on a 3D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(c, d, h, w)` multichannel 3D grid.
    u: torch.Tensor
        `(b, 3)` array of values in the range [0, 1].
        [0, 1] in b[:, 0] covers depth dim `d` of `grid`
        [0, 1] in b[:, 1] covers height dim `h` of `grid`
        [0, 1] in b[:, 2] covers width dim `w` of `grid`
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.
    monotonicity: str
        when either 'increasing' or 'decreasing' is specified, ensures
        that control points of spline are monotonic.

    Returns
    -------
    `(b, c)` array of c-dimensional interpolated values
    """
    if grid.ndim == 3:
        grid = einops.rearrange(grid, 'd h w -> 1 d h w')
    _, n_samples_d, n_samples_h, n_samples_w = grid.shape

    # expand grid to handle interpolation at edges
    grid = pad_grid_3d(grid)

    # find control point indices and interpolation coordinate in each dim
    idx_d, t_d = interpolants_to_interpolation_data_1d(u[:, 0], n_samples_d)
    idx_h, t_h = interpolants_to_interpolation_data_1d(u[:, 1], n_samples_h)
    idx_w, t_w = interpolants_to_interpolation_data_1d(u[:, 2], n_samples_w)

    # construct (4, 4, 4) grids of control points and 3D interpolant then interpolate
    idx_d = einops.repeat(idx_d, 'b d -> b d h w', h=4, w=4)
    idx_h = einops.repeat(idx_h, 'b h -> b d h w', d=4, w=4)
    idx_w = einops.repeat(idx_w, 'b w -> b d h w', d=4, h=4)
    if monotonicity:
        grid = transform_to_monotonic_nd(grid, ndims=3, monotonicity=monotonicity)
    control_points = grid[:, idx_d, idx_h, idx_w]  # (c, b, 4, 4, 4)
    control_points = einops.rearrange(control_points, 'c b d h w -> b c d h w')

    t = einops.rearrange([t_d, t_h, t_w], 'dhw b -> b dhw')
    return interpolate_pieces_3d(control_points, t, matrix=matrix)


def interpolate_grid_4d(
    grid: torch.Tensor,
    u: torch.Tensor,
    matrix: torch.Tensor,
    monotonicity: Optional[MonotonicityType] = None,
) -> torch.Tensor:
    """Uniform cubic B-spline interpolation on a 4D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(c, u, d, h, w)` multichannel 4D grid.
    u: torch.Tensor
        `(b, 4)` array of values in the range [0, 1].
        [0, 1] in b[:, 0] covers time dim `u` of `grid`
        [0, 1] in b[:, 1] covers depth dim `d` of `grid`
        [0, 1] in b[:, 2] covers height dim `h` of `grid`
        [0, 1] in b[:, 3] covers width dim `w` of `grid`
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.
    monotonicity: str
        when either 'increasing' or 'decreasing' is specified, ensures
        that control points of spline are monotonic.

    Returns
    -------
    `(b, c)` array of c-dimensional interpolated values
    """
    if grid.ndim == 4:
        grid = einops.rearrange(grid, 't d h w -> 1 t d h w')
    _, t, d, h, w = grid.shape

    # expand grid to handle interpolation at edges
    grid = pad_grid_4d(grid)

    # find control point indices and interpolation coordinate in each dim
    idx_t, t_t = interpolants_to_interpolation_data_1d(u[:, 0], n_samples=t)
    idx_d, t_d = interpolants_to_interpolation_data_1d(u[:, 1], n_samples=d)
    idx_h, t_h = interpolants_to_interpolation_data_1d(u[:, 2], n_samples=h)
    idx_w, t_w = interpolants_to_interpolation_data_1d(u[:, 3], n_samples=w)

    # construct (4, 4, 4, 4) grids of control points and 4D interpolant then interpolate
    idx_t = einops.repeat(idx_t, 'b t -> b t d h w', d=4, h=4, w=4)
    idx_d = einops.repeat(idx_d, 'b d -> b t d h w', t=4, h=4, w=4)
    idx_h = einops.repeat(idx_h, 'b h -> b t d h w', t=4, d=4, w=4)
    idx_w = einops.repeat(idx_w, 'b w -> b t d h w', t=4, d=4, h=4)
    if monotonicity:
        grid = transform_to_monotonic_nd(grid, ndims=3, monotonicity=monotonicity)
    control_points = grid[:, idx_t, idx_d, idx_h, idx_w]  # (c, b, 4, 4, 4, 4)
    control_points = einops.rearrange(control_points, 'c b t d h w -> b c t d h w')

    t = einops.rearrange([t_t, t_d, t_h, t_w], 'tdhw b -> b tdhw')
    return interpolate_pieces_4d(control_points, t, matrix=matrix)
