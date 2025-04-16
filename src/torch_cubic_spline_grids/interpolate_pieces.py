"""Interpolate 'pieces' for piecewise uniform cubic B-spline interpolation."""

import einops
import torch


def interpolate_pieces_1d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Batched uniform 1D cubic spline interpolation.

    ```
    [0, u, u^2, u^3] * [a00, a01, a02, a03]  * [p0]
                       [a10, a11, a12, a13]    [p1]
                       [a20, a21, a22, a23]    [p2]
                       [a30, a31, a32, a33]    [p3]
    ```
    c.f. Freya Holmer - "The Continuity of Splines": https://youtu.be/jvPPXbo87ds?t=3462

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, c, 4)` batch of 4 uniformly spaced control points `[p0, p1, p2, p3]`
        in `c` channels.
    t: torch.Tensor
        `(b, )` batch of interpolants in the range [0, 1] covering the interpolation
        interval between `p1` and `p2`
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.

    Returns
    -------
    interpolated: torch.Tensor
        `(b, c)` array of per-channel interpolants of `control_points` at `u`.
    """
    t = einops.rearrange([t**0, t, t**2, t**3], 'u b -> b 1 1 u')
    control_points = einops.rearrange(control_points, 'b c p -> b c p 1')
    interpolated = t @ matrix @ control_points
    return einops.rearrange(interpolated, 'b c 1 1 -> b c')


def interpolate_pieces_2d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Batched uniform 2D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, c, 4, 4)` batch of 2D multichannel grids of uniformly spaced control
        points `[p0, p1, p2, p3]` for cubic B-spline interpolation.
    t: torch.Tensor
        `(b, 2)` batch of values in the range `[0, 1]` defining the position of 2D
        points to be interpolated within the interval `[p1, p2]` along dim 1 and 2 of
        the 2D grid of control points.
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    # extract (b, c, 4) control points at each height along width dim of (h, w) grid
    h0, h1, h2, h3 = einops.rearrange(control_points, 'b c h w -> h b c w')

    # separate u into components along height and width dimensions
    t_h, t_w = einops.rearrange(t, 'b hw -> hw b')

    # 1d interpolation along width dim at each height
    p0 = interpolate_pieces_1d(control_points=h0, t=t_w, matrix=matrix)
    p1 = interpolate_pieces_1d(control_points=h1, t=t_w, matrix=matrix)
    p2 = interpolate_pieces_1d(control_points=h2, t=t_w, matrix=matrix)
    p3 = interpolate_pieces_1d(control_points=h3, t=t_w, matrix=matrix)

    # 1d interpolation of result along height dim
    control_points = einops.rearrange([p0, p1, p2, p3], 'p b c -> b c p')
    return interpolate_pieces_1d(control_points=control_points, t=t_h, matrix=matrix)


def interpolate_pieces_3d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Batched uniform 3D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, c, 4, 4, 4)` batch of `(4, 4, 4)` multichannel grids of uniformly
        spaced control points for cubic B-spline interpolation.
    t: torch.Tensor
        `(b, 3)` batch of values in the range `[0, 1]` defining the position of 3D
        points to be interpolated within the interval `[p1, p2]` along dim -3,
        -2 and -1 of `control_points`
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.

    Returns
    -------
    interpolated:
        `(b, c)` batch interpolated values in each channel.
    """
    # extract (b, c, 4, 4) 2D control point planes at each point along the depth dim
    d0, d1, d2, d3 = einops.rearrange(control_points, 'b c d h w -> d b c h w')

    # separate u into components along depth and (height, width) dimensions
    t_d = t[:, 0]
    t_hw = t[:, [1, 2]]

    # 2d interpolation on each (height, width) plane at each depth
    p0 = interpolate_pieces_2d(control_points=d0, t=t_hw, matrix=matrix)
    p1 = interpolate_pieces_2d(control_points=d1, t=t_hw, matrix=matrix)
    p2 = interpolate_pieces_2d(control_points=d2, t=t_hw, matrix=matrix)
    p3 = interpolate_pieces_2d(control_points=d3, t=t_hw, matrix=matrix)

    # 1d interpolation of result along depth dim
    control_points = einops.rearrange([p0, p1, p2, p3], 'p b c -> b c p')
    return interpolate_pieces_1d(control_points=control_points, t=t_d, matrix=matrix)


def interpolate_pieces_4d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor
) -> torch.Tensor:
    """Batched 4D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, c, 4, 4, 4, 4)` batch of multichannel `(4, 4, 4, 4)` grids of uniformly
        spaced control points for cubic B-spline interpolation.
    t: torch.Tensor
        `(b, 4)` batch of values in the range `[0, 1]` defining the position of 4D
        points to be interpolated within the interval `[p1, p2]` along dims -4, -3,
        -2 and -1 of `control_points`.
    matrix: torch.Tensor
        `(4, 4)` characteristic matrix for the spline.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    # extract (b, c, 4, 4, 4) 3D control point grids at each point along the time dim
    t0, t1, t2, t3 = einops.rearrange(control_points, 'b c u d h w -> u b c d h w')

    # separate u into components along time and (depth, height, width) dimensions
    t_t = t[:, 0]
    t_dhw = t[:, [1, 2, 3]]

    # 3D interpolation on each 3D grid along time dimension
    p0 = interpolate_pieces_3d(control_points=t0, t=t_dhw, matrix=matrix)
    p1 = interpolate_pieces_3d(control_points=t1, t=t_dhw, matrix=matrix)
    p2 = interpolate_pieces_3d(control_points=t2, t=t_dhw, matrix=matrix)
    p3 = interpolate_pieces_3d(control_points=t3, t=t_dhw, matrix=matrix)

    # 1d interpolation of result along time dim
    control_points = einops.rearrange([p0, p1, p2, p3], 'p b c -> b c p')
    return interpolate_pieces_1d(control_points=control_points, t=t_t, matrix=matrix)
