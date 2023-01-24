"""Interpolate 'pieces' for piecewise uniform cubic B-spline interpolation."""
import torch
import einops

from .constants import CUBIC_B_SPLINE_MATRIX


def interpolate_pieces_1d(control_points: torch.Tensor, u: torch.Tensor):
    """Batched uniform 1D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, c, 4)` batch of 4 uniformly spaced control points `[p0, p1, p2, p3]`
        in `c` channels.
    u: torch.Tensor
        `(b, )` batch of interpolants in the range [0, 1] covering the interpolation
        interval between `p1` and `p2`

    Returns
    -------
    interpolated: torch.Tensor
        `(b, c)` array of per-channel interpolants of `control_points` at `u`.
    """
    u = einops.rearrange([u ** 0, u, u ** 2, u ** 3], 'u b -> b 1 1 u')
    control_points = einops.rearrange(control_points, 'b c s -> b c s 1')
    interpolated = u @ CUBIC_B_SPLINE_MATRIX @ control_points
    return einops.rearrange(interpolated, 'b c 1 1 -> b c')


def interpolate_pieces_2d(control_points: torch.Tensor, u: torch.Tensor):
    """Batched uniform 2D cubic B-spline interpolation.

    Parameters
    ----------
    control_points:
        `(b, 4, 4, n)` batch of 2D grids of uniformly spaced n-dimensional control
        points for cubic B-spline interpolation.
    u:
        `(b, 2)` batch of values in the range `[0, 1]` defining the position of 2D points
        to be interpolated within the interval `[s1, s2]` along dim 1 and 2 of
        the 2D grid of control points.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    # extract (b, 4, n) control points at each height along width dim of (h, w) grid
    h0 = control_points[:, 0, :, :]
    h1 = control_points[:, 1, :, :]
    h2 = control_points[:, 2, :, :]
    h3 = control_points[:, 3, :, :]

    # separate t into components along height and width dimensions
    u_h, u_w = einops.rearrange(u, 'b hw -> hw b')

    # 1d interpolation along width dim at each height
    s0 = interpolate_pieces_1d(control_points=h0, u=u_w)
    s1 = interpolate_pieces_1d(control_points=h1, u=u_w)
    s2 = interpolate_pieces_1d(control_points=h2, u=u_w)
    s3 = interpolate_pieces_1d(control_points=h3, u=u_w)

    # 1d interpolation of result along height dim
    control_points = einops.rearrange([s0, s1, s2, s3], 's b n -> b s n')
    return interpolate_pieces_1d(control_points=control_points, u=u_h)


def interpolate_pieces_3d(control_points: torch.Tensor, u: torch.Tensor):
    """Batched uniform 3D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, 4, 4, 4, n)` batch of 3D grids of uniformly spaced n-dimensional control
        points for cubic B-spline interpolation.
    u: torch.Tensor
        `(b, 3)` batch of values in the range `[0, 1]` defining the position of 3D
        points to be interpolated within the interval `[s1, s2]` along dim 1,
        2 and 3 of the 3D grid of `control_points`.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    # extract (b, 4, 4, n) 2D control point planes at each point along the depth dim
    d0 = control_points[:, 0, :, :, :]
    d1 = control_points[:, 1, :, :, :]
    d2 = control_points[:, 2, :, :, :]
    d3 = control_points[:, 3, :, :, :]

    # separate t into components along depth and (height, width) dimensions
    u_d = u[:, 0]
    u_hw = u[:, 1:]

    # 2d interpolation on each (height, width) plane at each depth
    s0 = interpolate_pieces_2d(control_points=d0, u=u_hw)
    s1 = interpolate_pieces_2d(control_points=d1, u=u_hw)
    s2 = interpolate_pieces_2d(control_points=d2, u=u_hw)
    s3 = interpolate_pieces_2d(control_points=d3, u=u_hw)

    # 1d interpolation of result along depth dim
    control_points = einops.rearrange([s0, s1, s2, s3], 's b n -> b s n')
    return interpolate_pieces_1d(control_points=control_points, u=u_d)

def interpolate_pieces_4d(control_points: torch.Tensor, u: torch.Tensor):
    """Batched 4D cubic B-spline interpolation.

    Parameters
    ----------
    control_points: torch.Tensor
        `(b, 4, 4, 4, 4, n)` batch of 4D grids of uniformly spaced n-dimensional control
        points for cubic B-spline interpolation.
    u: torch.Tensor
        `(b, 4)` batch of values in the range `[0, 1]` defining the position of 4D
        points to be interpolated within the interval `[s1, s2]` along dim 1,
        2, 3 and 4 of the 4D grid of `control_points`.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    # extract (b, 4, 4, 4, n) 3D control point grids at each point along the depth dim
    t0, t1, t2, t3 = einops.rearrange(control_points, 'b t d h w n -> t b d h w n')

    # separate u into components along time and (depth, height, width) dimensions
    u_t = u[:, 0]
    u_dhw = u[:, 1:]

    # 3D interpolation on each 3D grid along time dimension
    s0 = interpolate_pieces_3d(control_points=t0, u=u_dhw)
    s1 = interpolate_pieces_3d(control_points=t1, u=u_dhw)
    s2 = interpolate_pieces_3d(control_points=t2, u=u_dhw)
    s3 = interpolate_pieces_3d(control_points=t3, u=u_dhw)

    # 1d interpolation of result along time dim
    control_points = einops.rearrange([s0, s1, s2, s3], 's b n -> b s n')
    return interpolate_pieces_1d(control_points=control_points, u=u_t)