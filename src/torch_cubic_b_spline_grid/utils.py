import einops
import torch


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
