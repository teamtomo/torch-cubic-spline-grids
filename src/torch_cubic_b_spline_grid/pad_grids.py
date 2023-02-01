import einops
import torch


def pad_grid_1d(grid: torch.Tensor):
    """Pad in the last dimension according to local gradients.

    e.g. [0, 1, 2] -> [-1, 0, 1, 2, 3]

    grid: torch.Tensor
        `(..., w)` array of values to be padded in last dimension.

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., w+2)` padded array.
    """
    # remove singleton dimension if necessary
    w = grid.shape[-1]
    if w == 1:
        grid = einops.repeat(grid, '... w -> ... (repeat w)', repeat=2)

    # find values for padding at each end of width dim
    start = grid[..., 0] - (grid[..., 1] - grid[..., 0])
    end = grid[..., -1] + (grid[..., -1] - grid[..., -2])

    # reintroduce width dim lost during indexing
    start = einops.rearrange(start, '... -> ... 1')
    end = einops.repeat(end, '... -> ... 1')
    return torch.cat([start, grid, end], dim=-1)


def pad_grid_2d(grid: torch.Tensor) -> torch.Tensor:
    """Pad a 2D grid of values according to local gradients.

    ```
    e.g.            [[-3, -2, -1, 0]
         [[0, 1]     [-1,  0,  1, 2]
          [2, 3]] -> [ 1,  2,  3, 4]
                     [ 3,  4,  5, 6]]
    ```

    Parameters
    ----------
    grid: torch.Tensor
        `(..., h, w)` array of values to be padded in height and width dimensions.

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., h+2, w+2)` padded array.
    """
    # remove singleton dimension if necessary
    h = grid.shape[-2]
    if h == 1:
        grid = einops.repeat(grid, '... h w -> ... (repeat h) w', repeat=2)
    grid = pad_grid_1d(grid)  # pad width dim (..., h, w+2)

    # find values for padding at each end of height dim
    h_start = grid[..., 0, :] - (grid[..., 1, :] - grid[..., 0, :])
    h_end = grid[..., -1, :] + (grid[..., -1, :] - grid[..., -2, :])

    # reintroduce height dim lost through indexing
    h_start = einops.rearrange(h_start, '... w -> ... 1 w')
    h_end = einops.rearrange(h_end, '... w -> ... 1 w')

    # pad height dim
    return torch.cat([h_start, grid, h_end], dim=-2)


def pad_grid_3d(grid: torch.Tensor) -> torch.Tensor:
    """Pad a 3D grid of values according to local gradients.

    Parameters
    ----------
    grid: torch.Tensor
        `(..., d, h, w)` array of values to be padded in depth, height and width
        dimensions.

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., d+2, h+2, w+2)` padded array.
    """
    # remove singleton dimension if necessary
    d = grid.shape[-3]
    if d == 1:
        grid = einops.repeat(grid, '... d h w -> ... (repeat d) h w', repeat=2)

    # pad in height and width dims
    grid = pad_grid_2d(grid)

    # find values for padding at each end of depth dim
    d_start = grid[..., 0, :, :] - (grid[..., 1, :, :] - grid[..., 0, :, :])
    d_end = grid[..., -1, :, :] + (grid[..., -1, :, :] - grid[..., -2, :, :])

    # reintroduce depth dim dropped by indexing
    d_start = einops.rearrange(d_start, '... h w -> ... 1 h w')
    d_end = einops.rearrange(d_end, '... h w -> ... 1 h w')
    return torch.cat([d_start, grid, d_end], dim=-3)


def pad_grid_4d(grid: torch.Tensor) -> torch.Tensor:
    """Pad a 4D grid of values according to local gradients.

    Parameters
    ----------
    grid: torch.Tensor
        `(..., t, d, h, w)` array of values to be padded in time, depth, height and
        width dimensions.

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., t+2, d+2, h+2, w+2)` grid
    """
    # remove singleton dimension if necessary
    t = grid.shape[-4]
    if t == 1:
        grid = einops.repeat(grid, '... t d h w -> ... (repeat t) d h w', repeat=2)

    # pad in height and width dims
    grid = pad_grid_3d(grid)  # (..., t, d+2, h+2, w+2)

    # find values for padding at each end of time dim
    dt_start = grid[..., 1, :, :, :] - grid[..., 0, :, :, :]
    t_start = grid[..., 0, :, :, :] - dt_start
    dt_end = grid[..., -1, :, :, :] - grid[..., -2, :, :, :]
    t_end = grid[..., -1, :, :, :] + dt_end

    # reintroduce time dim dropped by indexing
    t_start = einops.rearrange(t_start, '... d h w -> ... 1 d h w')
    t_end = einops.rearrange(t_end, '... d h w -> ... 1 d h w')
    return torch.cat([t_start, grid, t_end], dim=-4)
