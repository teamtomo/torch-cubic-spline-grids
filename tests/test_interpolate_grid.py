import torch

from torch_cubic_b_spline_grid import interpolate_grid


def test_interpolate_grid_1d():
    """Check that 1d interpolation works as expected."""
    grid = torch.tensor([0, 1, 2, 3, 4, 5]).float()
    u = torch.tensor([0.5])
    result = interpolate_grid.interpolate_grid_1d(grid, u)
    expected = torch.tensor([2.5])
    assert torch.allclose(result, expected)


def test_interpolate_grid_1d_approx():
    """Check that 1D interpolation approximates a function."""
    control_x = torch.linspace(0, 2 * torch.pi, steps=50)
    control_y = torch.sin(control_x)
    sample_x = torch.linspace(0, 1, steps=1000)
    sample_y = interpolate_grid.interpolate_grid_1d(control_y, sample_x).squeeze()
    ground_truth_y = torch.sin(sample_x * 2 * torch.pi)
    mean_absolute_error = torch.mean(torch.abs(sample_y - ground_truth_y))
    assert mean_absolute_error <= 0.01


def test_interpolate_grid_2d():
    """Check that 2D interpolation works."""
    grid = torch.tensor(
        [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11],
         [12, 13, 14, 15]]
    ).float()
    u = torch.tensor([0.5, 0.5]).view(1, 2)
    result = interpolate_grid.interpolate_grid_2d(grid, u)
    expected = torch.tensor([7.5])
    assert torch.allclose(result, expected)


def test_interpolate_grid_3d():
    """Check that 3D interpolation works."""
    grid = torch.tensor(
        [[[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10, 11],
          [12, 13, 14, 15]],
         [[16, 17, 18, 19],
          [20, 21, 22, 23],
          [24, 25, 26, 27],
          [28, 29, 30, 31]],
         [[32, 33, 34, 35],
          [36, 37, 38, 39],
          [40, 41, 42, 43],
          [44, 45, 46, 47]],
         [[48, 49, 50, 51],
          [52, 53, 54, 55],
          [56, 57, 58, 59],
          [60, 61, 62, 63]]],
    ).float()
    u = torch.tensor([[0.5, 0.5, 0.5]]).view(1, 3)
    result = interpolate_grid.interpolate_grid_3d(grid, u)
    expected = torch.tensor([31.5])
    assert torch.allclose(result, expected)