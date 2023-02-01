from typing import Tuple

import einops
import torch

from torch_cubic_b_spline_grid.pad_grids import pad_grid_1d


def generate_sample_positions_for_padded_grid_1d(n_samples: int) -> torch.Tensor:
    """Generate a 1D vector of sample coordinates for a padded grid.

    Coordinate system is [0, 1] covering each dimension, pre-padding.
    e.g. for 6 samples on a padded grid
        `[-0.333, 0, 0.333, 0.666, 1, 1.333]`


    Parameters
    ----------
    n_samples: int
        The number of samples on the grid prior to padding.

    Returns
    -------
    sample_coordinates: torch.Tensor
        The coordinates in the [0, 1] coordinate system of each sample on the
        padded grid.
    """
    du = 1 / (n_samples - 1)
    sample_coordinates = torch.linspace(-du, 1 + du, steps=n_samples + 2)

    # fix for numerical stability issues around 0 and 1
    # ensures valid control point indices are selected
    epsilon = 1e-6
    sample_coordinates[1] = 0 - epsilon
    sample_coordinates[-2] = 1 + epsilon
    return sample_coordinates


def find_control_point_idx_1d(sample_positions: torch.Tensor, query_points: torch.Tensor):
    """Find indices of four control points required for cubic interpolation.

    E.g. for sample positions `[0, 1, 2, 3, 4, 5]` and query point `2.5` the control
    point indices would be `[1, 2, 3, 4]` as `2.5` lies between `2` and `3`

    Parameters
    ----------
    sample_positions: torch.Tensor
        Monotonically increasing 1D array of sample positions.
    query_points: torch.Tensor
        `(b, )` array of query points for which control point indices.
    Returns
    -------
    control_point_idx: torch.Tensor
        `(b, 4)` array of indices for control points.
    """
    # find index of upper bound of interval for each query point
    sample_positions = sample_positions.contiguous()
    query_points = query_points.contiguous()
    iub_idx = torch.searchsorted(sample_positions, query_points, side='right')

    # generate (b, 4) array of indices of control points [s0, s1, s2, s3]
    # required for cubic interpolation
    s0_idx = iub_idx - 2
    s1_idx = iub_idx - 1
    s2_idx = iub_idx
    s3_idx = iub_idx + 1
    return einops.rearrange([s0_idx, s1_idx, s2_idx, s3_idx], 's b -> b s')


def interpolants_to_interpolation_data_1d(
    interpolants: torch.Tensor, n_samples: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Find the necessary data for piecewise cubic interpolation on a padded grid.

    Two pieces of data are required for piecewise cubic interpolation
    - four control points `[p0, p1, p2, p3]`
    - the interpolation coordinate

    The interpolation coordinate is a value in the range [0, 1] telling us how far into
    the interval `[p1, p2]` a query point is.

    This function returns the indices of the control points and the interpolation
    coordinate for a 1D grid. Returning the indices rather than the control points
    makes this more flexible for use in multidimensional grid interpolation which
    requires reusing and combining control point indices across dimensions.

    Parameters
    ----------
    interpolants: torch.Tensor
        `(b, )` batch of values in range [0, 1] covering the dimension being interpolated.
    n_samples: int
        The number of samples on the grid being interpolated (prior to padding).

    Returns
    -------
    control_point_idx, interpolation_coordinate: Tuple[torch.Tensor, torch.Tensor]
        The indices of control points `[p0, p1, p2, p3]` on a padded 1D grid and the
        interpolation coordinate associated with the interval `[p1, p2]`.
    """
    interpolants = torch.clamp(interpolants, min=0, max=1)
    if n_samples > 1:
        grid_u = generate_sample_positions_for_padded_grid_1d(n_samples)
        control_point_idx = find_control_point_idx_1d(grid_u, query_points=interpolants)
        u_p1 = grid_u[control_point_idx[:, 1]]
        du = 1 / (n_samples - 1)
        interpolation_coordinate = (interpolants - u_p1) / du
    else:
        control_point_idx = einops.repeat(
            torch.tensor([0, 1, 2, 3]), 'p -> b p', b=len(interpolants)
        )
        interpolation_coordinate = einops.repeat(
            torch.tensor([0.5]), '1 -> b', b=len(interpolants)
        )
    return control_point_idx, interpolation_coordinate


def coerce_to_multichannel_grid(grid: torch.Tensor, grid_ndim: int):
    """If missing, add a channel dimension to a multidimensional grid.

    e.g. for a 2D (h, w) grid
          `h w -> 1 h w`
        `c h w -> c h w`
    """
    grid_is_multichannel = grid.ndim == grid_ndim + 1
    grid_is_single_channel = grid.ndim == grid_ndim
    if grid_is_single_channel is False and grid_is_multichannel is False:
        raise ValueError(f'expected a {grid_ndim}D grid, got {grid.ndim}')
    if grid_is_single_channel:
        grid = einops.rearrange(grid, '... -> 1 ...')
    return grid


