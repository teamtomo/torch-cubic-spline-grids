"""Interpolate 'pieces' for piecewise uniform cubic B-spline interpolation."""

import einops
import torch


def interpolate_pieces_1d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor, 
    out: torch.Tensor = None, t_powers_buf: torch.Tensor = None
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
    out: torch.Tensor, optional
        `(b, c)` pre-allocated output buffer. If None, creates new tensor.
    t_powers_buf: torch.Tensor, optional
        `(b, 4)` pre-allocated buffer for t powers. If None, creates new tensor.

    Returns
    -------
    interpolated: torch.Tensor
        `(b, c)` array of per-channel interpolants of `control_points` at `u`.
    """
    batch_size, channels = control_points.shape[:2]
    device = control_points.device
    dtype = control_points.dtype
    
    # Pre-allocate output if not provided
    if out is None:
        out = torch.empty((batch_size, channels), device=device, dtype=dtype)
    
    # Pre-allocate t_powers buffer and fill explicitly
    if t_powers_buf is None:
        t_powers_buf = torch.empty((batch_size, 4), device=device, dtype=dtype)
    
    # Explicitly fill t_powers buffer in-place - no intermediate list/tensor creation
    t_powers_buf[:, 0] = 1.0  # t^0
    t_powers_buf[:, 1] = t    # t^1
    t_powers_buf[:, 2] = t**2 # t^2
    t_powers_buf[:, 3] = t**3 # t^3
    
    # Use einops for reshaping - creates views when possible
    t_powers = einops.rearrange(t_powers_buf, 'b p -> b 1 1 p')  # (b, 1, 1, 4)
    control_points_reshaped = einops.rearrange(control_points, 'b c p -> b c p 1')  # (b, c, 4, 1)
    
    # Matrix multiplication: (b, 1, 1, 4) @ (4, 4) @ (b, c, 4, 1) -> (b, c, 1, 1)
    interpolated = t_powers @ matrix @ control_points_reshaped
    
    # Extract result using einops and store in output buffer
    out[:] = einops.rearrange(interpolated, 'b c 1 1 -> b c')
    return out


def interpolate_pieces_2d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor,
    buffers: dict = None
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
    buffers: dict, optional
        Pre-allocated buffers for intermediate computations to reduce memory allocation.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    batch_size, channels = control_points.shape[:2]
    device = control_points.device
    dtype = control_points.dtype
    
    # Initialize buffers if not provided
    if buffers is None:
        buffers = {}
    
    # Pre-allocate reusable buffers
    if 'width_results' not in buffers:
        buffers['width_results'] = torch.empty((4, batch_size, channels), device=device, dtype=dtype)
    if 'final_control_points' not in buffers:
        buffers['final_control_points'] = torch.empty((batch_size, channels, 4), device=device, dtype=dtype)
    if 'temp_1d' not in buffers:
        buffers['temp_1d'] = torch.empty((batch_size, channels), device=device, dtype=dtype)
    if 't_powers_buf' not in buffers:
        buffers['t_powers_buf'] = torch.empty((batch_size, 4), device=device, dtype=dtype)
    
    # Extract components more efficiently using tensor indexing
    t_h, t_w = t[:, 0], t[:, 1]
    
    # Batch process all 4 width interpolations at once
    width_results = buffers['width_results']
    temp_1d_buf = buffers['temp_1d']
    t_powers_buf = buffers['t_powers_buf']
    
    for i in range(4):
        # control_points[:, :, i, :] is (b, c, 4) - control points at height i
        interpolate_pieces_1d(
            control_points=control_points[:, :, i, :], 
            t=t_w, 
            matrix=matrix,
            out=temp_1d_buf,
            t_powers_buf=t_powers_buf
        )
        width_results[i] = temp_1d_buf
    
    # Prepare for height interpolation using einops - creates view when possible
    final_control_points = buffers['final_control_points']
    final_control_points[:] = einops.rearrange(width_results, 'p b c -> b c p')
    
    # Final interpolation along height dimension
    return interpolate_pieces_1d(
        control_points=final_control_points, 
        t=t_h, 
        matrix=matrix,
        out=temp_1d_buf,
        t_powers_buf=t_powers_buf
    )


def interpolate_pieces_3d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor,
    buffers: dict = None
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
    buffers: dict, optional
        Pre-allocated buffers for intermediate computations to reduce memory allocation.

    Returns
    -------
    interpolated:
        `(b, c)` batch interpolated values in each channel.
    """
    batch_size, channels = control_points.shape[:2]
    device = control_points.device
    dtype = control_points.dtype
    
    # Initialize buffers if not provided
    if buffers is None:
        buffers = {}
    
    # Pre-allocate reusable buffers for 3D
    if 'depth_results' not in buffers:
        buffers['depth_results'] = torch.empty((4, batch_size, channels), device=device, dtype=dtype)
    if 'final_control_points_3d' not in buffers:
        buffers['final_control_points_3d'] = torch.empty((batch_size, channels, 4), device=device, dtype=dtype)
    if 'temp_2d' not in buffers:
        buffers['temp_2d'] = torch.empty((batch_size, channels), device=device, dtype=dtype)
    if 't_powers_buf_3d' not in buffers:
        buffers['t_powers_buf_3d'] = torch.empty((batch_size, 4), device=device, dtype=dtype)
    
    # Create nested buffer dict for 2D operations
    if 'buffers_2d' not in buffers:
        buffers['buffers_2d'] = {}
    
    # Extract components
    t_d = t[:, 0]
    t_hw = t[:, [1, 2]]
    
    # Process all 4 depth planes
    depth_results = buffers['depth_results']
    temp_2d_buf = buffers['temp_2d']
    buffers_2d = buffers['buffers_2d']
    t_powers_buf = buffers['t_powers_buf_3d']
    
    for i in range(4):
        # control_points[:, :, i, :, :] is (b, c, 4, 4) - 2D plane at depth i
        result = interpolate_pieces_2d(
            control_points=control_points[:, :, i, :, :], 
            t=t_hw, 
            matrix=matrix,
            buffers=buffers_2d
        )
        depth_results[i] = result
    
    # Prepare for depth interpolation using einops - creates view when possible
    final_control_points = buffers['final_control_points_3d']
    final_control_points[:] = einops.rearrange(depth_results, 'p b c -> b c p')
    
    # Final interpolation along depth dimension
    return interpolate_pieces_1d(
        control_points=final_control_points, 
        t=t_d, 
        matrix=matrix,
        out=temp_2d_buf,
        t_powers_buf=t_powers_buf
    )


def interpolate_pieces_4d(
    control_points: torch.Tensor, t: torch.Tensor, matrix: torch.Tensor,
    buffers: dict = None
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
    buffers: dict, optional
        Pre-allocated buffers for intermediate computations to reduce memory allocation.

    Returns
    -------
    interpolated:
        `(b, n)` batch of n-dimensional interpolated values.
    """
    batch_size, channels = control_points.shape[:2]
    device = control_points.device
    dtype = control_points.dtype
    
    # Initialize buffers if not provided
    if buffers is None:
        buffers = {}
    
    # Pre-allocate reusable buffers for 4D
    if 'time_results' not in buffers:
        buffers['time_results'] = torch.empty((4, batch_size, channels), device=device, dtype=dtype)
    if 'final_control_points_4d' not in buffers:
        buffers['final_control_points_4d'] = torch.empty((batch_size, channels, 4), device=device, dtype=dtype)
    if 'temp_3d' not in buffers:
        buffers['temp_3d'] = torch.empty((batch_size, channels), device=device, dtype=dtype)
    if 't_powers_buf_4d' not in buffers:
        buffers['t_powers_buf_4d'] = torch.empty((batch_size, 4), device=device, dtype=dtype)
    
    # Create nested buffer dict for 3D operations
    if 'buffers_3d' not in buffers:
        buffers['buffers_3d'] = {}
    
    # Extract components
    t_t = t[:, 0]
    t_dhw = t[:, [1, 2, 3]]
    
    # Process all 4 time slices
    time_results = buffers['time_results']
    temp_3d_buf = buffers['temp_3d']
    buffers_3d = buffers['buffers_3d']
    t_powers_buf = buffers['t_powers_buf_4d']
    
    for i in range(4):
        # control_points[:, :, i, :, :, :] is (b, c, 4, 4, 4) - 3D grid at time i
        result = interpolate_pieces_3d(
            control_points=control_points[:, :, i, :, :, :], 
            t=t_dhw, 
            matrix=matrix,
            buffers=buffers_3d
        )
        time_results[i] = result
    
    # Prepare for time interpolation using einops - creates view when possible
    final_control_points = buffers['final_control_points_4d']
    final_control_points[:] = einops.rearrange(time_results, 'p b c -> b c p')
    
    # Final interpolation along time dimension
    return interpolate_pieces_1d(
        control_points=final_control_points, 
        t=t_t, 
        matrix=matrix,
        out=temp_3d_buf,
        t_powers_buf=t_powers_buf
    )
