def pad_grid_1d(grid: torch.Tensor):
    """Pad an array of vectors in dim -1 according to local gradients at each end.

    e.g. [0, 1, 2] -> [-1, 0, 1, 2, 3]

    grid: torch.Tensor
        `(..., w, n)` length w vector of n-dimensional values to be padded

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., w+2, n)`
    """
    start = grid[..., 0, :] - (grid[..., 1, :] - grid[..., 0, :])
    end = grid[..., -1, :] + (grid[..., -1, :] - grid[..., -2, :])

    # reintroduce width dim lost during indexing
    start = einops.rearrange(start, '... n -> ... 1 n')
    end = einops.repeat(end, '... n -> ... 1 n')
    return torch.cat([start, grid, end], dim=-2)


def pad_grid_2d(grid: torch.Tensor) -> torch.Tensor:
    """

    Parameters
    ----------
    grid: torch.Tensor
        `(..., h, w, n)` 2D grid which should be interpreted as a
        `(..., h, w)` array of n-dimensional values

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., h+2, w+2, n)` grid
    """
    grid = _pad_grid_1d(grid)  # pad width dim (..., h, w+2, n)

    # pad height dim
    h_start = grid[..., 0, :, :] - (grid[..., 1, :, :] - grid[..., 0, :, :])
    h_end = grid[..., -1, :, :] + (grid[..., -1, :, :] - grid[..., -2, :, :])

    # reintroduce height dim lost through indexing
    h_start = einops.rearrange(h_start, '... w d -> ... 1 w d')
    h_end = einops.rearrange(h_end, '... w d -> ... 1 w d')

    # pad height dim
    return torch.cat([h_start, grid, h_end], dim=-3)


def pad_grid_3d(grid: torch.Tensor) -> torch.Tensor:
    """

    Parameters
    ----------
    grid: torch.Tensor
        `(..., d, h, w, n)` 3D grid which should be interpreted as a
        `(..., d, h, w)` array of n-dimensional values.

    Returns
    -------
    padded_grid: torch.Tensor
        `(d+2, h+2, w+2, n)` grid
    """
    # pad in height and width dims
    grid = _pad_grid_2d(grid)

    # pad in depth dim
    d_start = grid[..., 0, :, :, :] - (grid[..., 1, :, :, :] - grid[..., 0, :, :, :])
    d_end = grid[..., -1, :, :, :] + (grid[..., -1, :, :, :] - grid[..., -2, :, :, :])

    # reintroduce depth dim dropped by indexing
    d_start = einops.rearrange(d_start, '... h w n -> ... 1 h w n')
    d_end = einops.rearrange(d_end, '... h w n -> ... 1 h w n')
    return torch.cat([d_start, grid, d_end])


def pad_grid_4d(grid: torch.Tensor) -> torch.Tensor:
    """

    Parameters
    ----------
    grid: torch.Tensor
        `(..., t, d, h, w, n)` 4+D grid which should be interpreted as a
        `(..., t, d, h, w)` array of n-dimensional values.

    Returns
    -------
    padded_grid: torch.Tensor
        `(..., t+2, d+2, h+2, w+2, n)` grid
    """
    # pad in height and width dims
    grid = _pad_grid_3d(grid)

    # pad in time dim
    t_start = grid[..., 0, :, , :, :] - (grid[..., 1, :, , :, :] - grid[..., 0, :, :,
                                                                   :, :])
    t_end = grid[..., -1, :, :, :] + (grid[..., -1, :, :, :] - grid[..., -2, :, :, :])

    # reintroduce time dim dropped by indexing
    t_start = einops.rearrange(t_start, '... d h w n -> ... 1 d h w n')
    t_end = einops.rearrange(t_end, '... d h w n -> ... 1 d h w n')
    return torch.cat([t_start, grid, t_end])