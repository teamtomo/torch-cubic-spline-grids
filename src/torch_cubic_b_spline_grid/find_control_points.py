import einops
import torch


def find_control_points_1d(sample_positions: torch.Tensor, query_points: torch.Tensor):
    """Find indices of the four control points required for cubic interpolation.

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
    # fix for numerical stability issues around 0 and 1
    # ensures valid control point indices are selected
    epsilon = 1e-6
    sample_positions[sample_positions < epsilon] = 0 - epsilon
    sample_positions[torch.abs(sample_positions - 1) < epsilon] = 1 + epsilon

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