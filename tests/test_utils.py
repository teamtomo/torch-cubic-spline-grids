import einops
import torch

from torch_cubic_spline_grids.utils import find_control_point_idx_1d, batch


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


def test_batch():
    """All items should be present in minibatches."""
    l = [0, 1, 2, 3, 4, 5, 6]
    minibatches = [minibatch for minibatch in batch(l, n=3)]
    expected = [[0, 1, 2], [3, 4, 5], [6]]
    assert minibatches == expected


def test_restacking_batch():
    """Ensure entries get restacked by cat the same way as they are unstacked."""
    batched_input = torch.rand(size=(10, 3))  # (b, d)
    minibatches = [minibatch for minibatch in batch(batched_input, n=3)]
    restacked_minibatches = torch.cat(minibatches, dim=0)
    assert torch.allclose(batched_input, restacked_minibatches)
