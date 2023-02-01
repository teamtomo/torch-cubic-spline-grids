import torch

from torch_cubic_b_spline_grid import interpolate_grids


def test_interpolate_grid_1d():
    """Check that 1d interpolation works as expected."""
    grid = torch.tensor([0, 1, 2, 3, 4, 5]).float()
    u = torch.tensor([0.5])
    result = interpolate_grids.interpolate_grid_1d(grid, u)
    expected = torch.tensor([2.5])
    assert torch.allclose(result, expected)


def test_interpolate_grid_1d_approx():
    """Check that 1D interpolation approximates a function."""
    control_x = torch.linspace(0, 2 * torch.pi, steps=50)
    control_y = torch.sin(control_x)
    sample_x = torch.linspace(0, 1, steps=1000)
    sample_y = interpolate_grids.interpolate_grid_1d(control_y, sample_x).squeeze()
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
    result = interpolate_grids.interpolate_grid_2d(grid, u)
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
    result = interpolate_grids.interpolate_grid_3d(grid, u)
    expected = torch.tensor([31.5])
    assert torch.allclose(result, expected)


def test_interpolate_grid_4d():
    """Check that 4D interpolation works as expected."""
    grid = torch.tensor(
        [[[[0, 1, 2, 3],
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
         [[[64, 65, 66, 67],
           [68, 69, 70, 71],
           [72, 73, 74, 75],
           [76, 77, 78, 79]],
          [[80, 81, 82, 83],
           [84, 85, 86, 87],
           [88, 89, 90, 91],
           [92, 93, 94, 95]],
          [[96, 97, 98, 99],
           [100, 101, 102, 103],
           [104, 105, 106, 107],
           [108, 109, 110, 111]],
          [[112, 113, 114, 115],
           [116, 117, 118, 119],
           [120, 121, 122, 123],
           [124, 125, 126, 127]]],
         [[[128, 129, 130, 131],
           [132, 133, 134, 135],
           [136, 137, 138, 139],
           [140, 141, 142, 143]],
          [[144, 145, 146, 147],
           [148, 149, 150, 151],
           [152, 153, 154, 155],
           [156, 157, 158, 159]],
          [[160, 161, 162, 163],
           [164, 165, 166, 167],
           [168, 169, 170, 171],
           [172, 173, 174, 175]],
          [[176, 177, 178, 179],
           [180, 181, 182, 183],
           [184, 185, 186, 187],
           [188, 189, 190, 191]]],
         [[[192, 193, 194, 195],
           [196, 197, 198, 199],
           [200, 201, 202, 203],
           [204, 205, 206, 207]],
          [[208, 209, 210, 211],
           [212, 213, 214, 215],
           [216, 217, 218, 219],
           [220, 221, 222, 223]],
          [[224, 225, 226, 227],
           [228, 229, 230, 231],
           [232, 233, 234, 235],
           [236, 237, 238, 239]],
          [[240, 241, 242, 243],
           [244, 245, 246, 247],
           [248, 249, 250, 251],
           [252, 253, 254, 255]]]]
    ).float()
    u = torch.tensor([0.5, 0.5, 0.5, 0.5]).view(1, 4)
    result = interpolate_grids.interpolate_grid_4d(grid, u)
    expected = torch.tensor([127.5])
    assert torch.allclose(result, expected)
