import torch

from torch_cubic_b_spline_grid.utils import find_control_point_idx_1d


def test_find_control_points():
    sample_positions = torch.tensor([0, 1, 2, 3, 4, 5, 6])

    # sample between points should yield four closest points
    result = find_control_point_idx_1d(sample_positions, torch.tensor([2.5]))
    expected = torch.tensor([[1, 2, 3, 4]])
    assert torch.allclose(result, expected)

    # sample on point should be included as lower bound in interval
    result = find_control_point_idx_1d(sample_positions, torch.tensor([2]))
    expected = torch.tensor([[1, 2, 3, 4]])
    assert torch.allclose(result, expected)

    # check the same is true for 3, the upper bound of the same interval
    result = find_control_point_idx_1d(sample_positions, torch.tensor([3]))
    expected = torch.tensor([[2, 3, 4, 5]])
    assert torch.allclose(result, expected)
