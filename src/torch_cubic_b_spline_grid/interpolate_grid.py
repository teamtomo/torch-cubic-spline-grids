import einops
import torch

from torch_cubic_b_spline_grid.pad_grid import (
    pad_grid_1d,
    pad_grid_2d,
    pad_grid_3d,
    pad_grid_4d,
)
from torch_cubic_b_spline_grid.find_control_points import find_control_points_1d
from torch_cubic_b_spline_grid.interpolate_pieces import (
    interpolate_pieces_1d,
    interpolate_pieces_2d,
    interpolate_pieces_3d,
    interpolate_pieces_4d,
)


def interpolate_grid_1d(grid: torch.Tensor, u: torch.Tensor):
    """Uniform cubic B-spline interpolation on a 1D grid.

    The range [0, 1] covers all data points in the 1D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(w, c)` length `w` vector of values in `c` channels to be interpolated.
    u: torch.Tensor
        `(b, )` array of query points in the range `[0, 1]` covering the `w`
        dimension of `grid`.
    Returns
    -------
    interpolated: torch.Tensor
        `(b, c)` array of interpolated values in each channel..
    """
    if grid.ndim == 1:
        grid = einops.rearrange(grid, 'w -> 1 w')
    epsilon = 1e-6
    u[u == 1] -= epsilon
    n_samples = len(grid)

    # handle interpolation at edges by extending grid of control points according to
    # local gradients
    grid = pad_grid_1d(grid)

    # find the correct four control points for each query point in u
    du = 1 / (n_samples - 1)
    grid_positions = torch.linspace(-du, 1 + du, steps=n_samples + 2)
    control_point_idx = find_control_points_1d(
        sample_positions=grid_positions, query_points=u
    )
    control_points = grid[control_point_idx]

    # how far into the interpolation interval is each query point?
    s1_idx = control_point_idx[:, 1]
    u_s1 = grid_positions[s1_idx]
    interpolation_u = (u - u_s1) / du

    # interpolate
    return interpolate_pieces_1d(control_points, interpolation_u)


def interpolate_grid_2d(grid: torch.Tensor, u: torch.Tensor):
    """Uniform cubic B-spline interpolation on a 2D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(h, w, c)` 2D grid of uniformly spaced values in `c` channels to be
        interpolated.
    u: torch.Tensor
        `(b, 2)` array of values in the range [0, 1].
        [0, 1] in b[:, 0] covers dim 0 (h) of y
        [0, 1] in b[:, 1] covers dim 1 (w) of y

    Returns
    -------
    `(b, c)` array of interpolated values in each channel.
    """
    if grid.ndim == 2:
        grid = einops.rearrange(grid, 'h w -> h w 1')
    n_samples_h, n_samples_w, _ = grid.shape

    # pad grid to handle interpolation at edges.
    grid = pad_grid_2d(grid)

    # find indices for the four control points in each dimension
    du_h = 1 / (n_samples_h - 1)
    du_w = 1 / (n_samples_w - 1)
    grid_u_h = torch.linspace(-du_h, 1 + du_h, steps=n_samples_h + 2)
    grid_u_w = torch.linspace(-du_w, 1 + du_w, steps=n_samples_w + 2)
    control_point_idx_h = find_control_points_1d(
        sample_positions=grid_u_h, query_points=u[:, 0]
    )
    control_point_idx_w = find_control_points_1d(
        sample_positions=grid_u_w, query_points=u[:, 1]
    )

    # how far into the interpolation interval is each query point along each dimension
    s1_h_idx = control_point_idx_h[:, 1]
    s1_w_idx = control_point_idx_w[:, 1]
    s1_h = grid_u_h[s1_h_idx]
    s1_w = grid_u_w[s1_w_idx]
    interpolation_u_h = (u[:, 0] - s1_h) / du_h
    interpolation_u_w = (u[:, 1] - s1_w) / du_w
    interpolation_u = einops.rearrange(
        [interpolation_u_h, interpolation_u_w], 'hw b -> b hw'
    )

    # construct (4, 4) grids of control points for 2D interpolation and interpolate
    control_point_grid_idx = (
        einops.repeat(control_point_idx_h, 'b h -> b h w', w=4),
        einops.repeat(control_point_idx_w, 'b w -> b h w', h=4)
    )
    control_points = grid[control_point_grid_idx]  # (b, 4, 4, d)
    return interpolate_pieces_2d(control_points, interpolation_u)


def interpolate_grid_3d(grid, u):
    """Uniform cubic B-spline interpolation on a 3D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(d, h, w, c)` 3D grid of c-dimensional points.
    u: torch.Tensor
        `(b, 3)` array of values in the range [0, 1].
        [0, 1] in b[:, 0] covers depth dim `d` of `grid`
        [0, 1] in b[:, 1] covers height dim `h` of `grid`
        [0, 1] in b[:, 2] covers width dim `w` of `grid`

    Returns
    -------
    `(b, c)` array of c-dimensional interpolated values
    """
    if grid.ndim == 3:
        grid = einops.rearrange(grid, 'd h w -> d h w 1')
    n_samples_d, n_samples_h, n_samples_w, _ = grid.shape

    # expand grid to handle interpolation at edges
    grid = pad_grid_3d(grid)

    # find indices for control points in each dimension
    dd = 1 / (n_samples_d - 1)
    dh = 1 / (n_samples_h - 1)
    dw = 1 / (n_samples_w - 1)
    grid_u_d = torch.linspace(-dd, 1 + dd, steps=n_samples_d + 2)
    grid_u_h = torch.linspace(-dh, 1 + dh, steps=n_samples_h + 2)
    grid_u_w = torch.linspace(-dw, 1 + dw, steps=n_samples_w + 2)
    control_point_idx_d = find_control_points_1d(
        sample_positions=grid_u_d, query_points=u[:, 0]
    )
    control_point_idx_h = find_control_points_1d(
        sample_positions=grid_u_h, query_points=u[:, 1]
    )
    control_point_idx_w = find_control_points_1d(
        sample_positions=grid_u_w, query_points=u[:, 2]
    )

    # how far into the interpolation interval is each query point
    s1_idx_d = control_point_idx_d[:, 1]
    s1_idx_h = control_point_idx_h[:, 1]
    s1_idx_w = control_point_idx_w[:, 1]
    s1_d = grid_u_d[s1_idx_d]
    s1_h = grid_u_h[s1_idx_h]
    s1_w = grid_u_w[s1_idx_w]
    interpolation_u_d = (u[:, 0] - s1_d) / dd
    interpolation_u_h = (u[:, 1] - s1_h) / dh
    interpolation_u_w = (u[:, 2] - s1_w) / dw
    interpolation_u = einops.rearrange(
        [interpolation_u_d, interpolation_u_h, interpolation_u_w], 'dhw b -> b dhw'
    )

    # grid the control point indices and interpolate
    control_point_grid_idx = (
        einops.repeat(control_point_idx_d, 'b d -> b d h w', h=4, w=4),
        einops.repeat(control_point_idx_h, 'b h -> b d h w', d=4, w=4),
        einops.repeat(control_point_idx_w, 'b w -> b d h w', d=4, h=4),
    )
    control_points = grid[control_point_grid_idx]  # (b, 4, 4, 4, c)
    return interpolate_pieces_3d(control_points, interpolation_u)


def interpolate_grid_4d(grid: torch.Tensor, u: torch.Tensor):
    """Uniform cubic B-spline interpolation on a 4D grid.

    Parameters
    ----------
    grid: torch.Tensor
        `(t, d, h, w, c)` 4D grid of c-dimensional points.
    u: torch.Tensor
        `(b, 4)` array of values in the range [0, 1].
        [0, 1] in b[:, 0] covers time dim `t` of `grid`
        [0, 1] in b[:, 1] covers depth dim `d` of `grid`
        [0, 1] in b[:, 2] covers height dim `h` of `grid`
        [0, 1] in b[:, 3] covers width dim `w` of `grid`

    Returns
    -------
    `(b, c)` array of c-dimensional interpolated values
    """
    if grid.ndim == 4:
        grid = einops.rearrange(grid, 't d h w -> t d h w 1')
    n_samples_t, n_samples_d, n_samples_h, n_samples_w, _ = grid.shape

    # expand grid to handle interpolation at edges
    grid = pad_grid_4d(grid)

    # find indices for control points in each dimension
    dt = 1 / (n_samples_t - 1)
    dd = 1 / (n_samples_d - 1)
    dh = 1 / (n_samples_h - 1)
    dw = 1 / (n_samples_w - 1)

    grid_u_t = torch.linspace(-dt, 1 + dt, steps=n_samples_t + 2)
    grid_u_d = torch.linspace(-dd, 1 + dd, steps=n_samples_d + 2)
    grid_u_h = torch.linspace(-dh, 1 + dh, steps=n_samples_h + 2)
    grid_u_w = torch.linspace(-dw, 1 + dw, steps=n_samples_w + 2)
    control_point_idx_t = find_control_points_1d(
        sample_positions=grid_u_t, query_points=u[:, 0]
    )
    control_point_idx_d = find_control_points_1d(
        sample_positions=grid_u_d, query_points=u[:, 1]
    )
    control_point_idx_h = find_control_points_1d(
        sample_positions=grid_u_h, query_points=u[:, 2]
    )
    control_point_idx_w = find_control_points_1d(
        sample_positions=grid_u_w, query_points=u[:, 3]
    )

    # how far into the interpolation interval is each query point
    s1_idx_t = control_point_idx_t[:, 1]
    s1_idx_d = control_point_idx_d[:, 1]
    s1_idx_h = control_point_idx_h[:, 1]
    s1_idx_w = control_point_idx_w[:, 1]
    s1_t = grid_u_t[s1_idx_t]
    s1_d = grid_u_d[s1_idx_d]
    s1_h = grid_u_h[s1_idx_h]
    s1_w = grid_u_w[s1_idx_w]
    interpolation_u_t = (u[:, 0] - s1_t) / dt
    interpolation_u_d = (u[:, 1] - s1_d) / dd
    interpolation_u_h = (u[:, 2] - s1_h) / dh
    interpolation_u_w = (u[:, 3] - s1_w) / dw
    interpolation_u = [
        interpolation_u_t,
        interpolation_u_d,
        interpolation_u_h,
        interpolation_u_w
    ]
    interpolation_u = einops.rearrange(interpolation_u, 'tdhw b -> b tdhw')

    # grid the control point indices and interpolate
    control_point_grid_idx = (
        einops.repeat(control_point_idx_t, 'b t -> b t d h w', d=4, h=4, w=4),
        einops.repeat(control_point_idx_d, 'b d -> b t d h w', t=4, h=4, w=4),
        einops.repeat(control_point_idx_h, 'b h -> b t d h w', t=4, d=4, w=4),
        einops.repeat(control_point_idx_w, 'b w -> b t d h w', t=4, d=4, h=4),
    )
    control_points = grid[control_point_grid_idx]  # (b, 4, 4, 4, 4, c)
    return interpolate_pieces_4d(control_points, interpolation_u)