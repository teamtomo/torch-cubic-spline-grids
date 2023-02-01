import torch

from torch_cubic_b_spline_grid import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
)


def test_1d_grid_direct_instantiation():
    """Test grid instantiation with different types for resolution argument."""
    grid = CubicBSplineGrid1d()
    assert isinstance(grid, CubicBSplineGrid1d)
    assert grid.data.shape == (1, 2)

    grid = CubicBSplineGrid1d(resolution=5, n_channels=3)
    assert isinstance(grid, CubicBSplineGrid1d)
    assert grid.data.shape == (3, 5)

    grid = CubicBSplineGrid1d(resolution=(5,), n_channels=3)
    assert isinstance(grid, CubicBSplineGrid1d)
    assert grid.data.shape == (3, 5)


def test_1d_grid_instantiation_from_existing_data():
    """Test grid instantiation from existing data."""
    grid = CubicBSplineGrid1d.from_grid_data(data=torch.zeros(3, 5))
    assert grid.ndim == 1
    assert grid.resolution == (5,)
    assert grid.n_channels == 3
    assert isinstance(grid._data, torch.nn.Parameter)


def test_calling_1d_grid():
    """Test calling 1d grid."""
    grid = CubicBSplineGrid1d()
    expected = torch.tensor([0.])
    for arg in (0.5, [0.5], torch.tensor([0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


def test_1d_grid_with_singleton_dimension():
    """Test that a 2D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = CubicBSplineGrid1d(resolution=1)
    result = grid(0.5)
    assert torch.allclose(result, torch.tensor([0.0]))


def test_2d_grid_direct_instantiation():
    grid = CubicBSplineGrid2d()
    assert isinstance(grid, CubicBSplineGrid2d)
    assert grid.data.shape == (1, 2, 2)

    grid = CubicBSplineGrid2d(resolution=(5, 4), n_channels=3)
    assert isinstance(grid, CubicBSplineGrid2d)
    assert grid.data.shape == (3, 5, 4)


def test_2d_grid_instantiation_from_existing_data():
    """Test grid instantiation from existing data."""
    grid = CubicBSplineGrid2d.from_grid_data(data=torch.zeros(3, 5, 4))
    assert grid.ndim == 2
    assert grid.resolution == (5, 4)
    assert grid.n_channels == 3
    assert isinstance(grid._data, torch.nn.Parameter)


def test_calling_2d_grid():
    """Test calling 2d grid."""
    grid = CubicBSplineGrid2d()
    expected = torch.tensor([0., 0.])
    for arg in ([0.5, 0.5], torch.tensor([0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


def test_2d_grid_with_singleton_dimension():
    """Test that a 2D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = CubicBSplineGrid2d(resolution=(2, 1))
    result = grid([0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0]))

    # singleton in height dim
    grid = CubicBSplineGrid2d(resolution=(1, 2))
    result = grid([0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0]))


def test_3d_grid_direct_instantiation():
    grid = CubicBSplineGrid3d()
    assert isinstance(grid, CubicBSplineGrid3d)
    assert grid.data.shape == (1, 2, 2, 2)

    grid = CubicBSplineGrid3d(resolution=(5, 4, 3), n_channels=2)
    assert isinstance(grid, CubicBSplineGrid3d)
    assert grid.data.shape == (2, 5, 4, 3)


def test_3d_grid_instantiation_from_existing_data():
    """Test grid instantiation from existing data."""
    grid = CubicBSplineGrid3d.from_grid_data(data=torch.zeros(2, 5, 4, 3))
    assert grid.ndim == 3
    assert grid.resolution == (5, 4, 3)
    assert grid.n_channels == 2
    assert isinstance(grid._data, torch.nn.Parameter)


def test_calling_3d_grid():
    """Test calling 3d grid."""
    grid = CubicBSplineGrid3d()
    expected = torch.tensor([0., 0., 0.])
    for arg in ([0.5, 0.5, 0.5], torch.tensor([0.5, 0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


def test_3d_grid_with_singleton_dimension():
    """Test that a 3D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = CubicBSplineGrid3d(resolution=(2, 2, 1))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    # singleton in height dim
    grid = CubicBSplineGrid3d(resolution=(2, 1, 2))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    # singleton in depth dim
    grid = CubicBSplineGrid3d(resolution=(1, 2, 2))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))


def test_4d_grid_direct_instantiation():
    grid = CubicBSplineGrid4d()
    assert isinstance(grid, CubicBSplineGrid4d)
    assert grid.data.shape == (1, 2, 2, 2, 2)

    grid = CubicBSplineGrid4d(resolution=(6, 5, 4, 3), n_channels=2)
    assert isinstance(grid, CubicBSplineGrid4d)
    assert grid.data.shape == (2, 6, 5, 4, 3)


def test_4d_grid_instantiation_from_existing_data():
    """Test grid instantiation from existing data."""
    grid = CubicBSplineGrid4d.from_grid_data(data=torch.zeros(2, 6, 5, 4, 3))
    assert grid.ndim == 4
    assert grid.resolution == (6, 5, 4, 3)
    assert grid.n_channels == 2
    assert isinstance(grid._data, torch.nn.Parameter)


def test_calling_4d_grid():
    """Test calling 3d grid."""
    grid = CubicBSplineGrid4d()
    expected = torch.tensor([0., 0., 0., 0.])
    for arg in ([0.5, 0.5, 0.5, 0.5], torch.tensor([0.5, 0.5, 0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


def test_4d_grid_with_singleton_dimension():
    """Test that a 4D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = CubicBSplineGrid4d(resolution=(2, 2, 2, 1))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in height dim
    grid = CubicBSplineGrid4d(resolution=(2, 2, 1, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in depth dim
    grid = CubicBSplineGrid4d(resolution=(2, 1, 2, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in time dim
    grid = CubicBSplineGrid4d(resolution=(1, 2, 2, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # multiple singletons
    grid = CubicBSplineGrid4d(resolution=(1, 1, 1, 1))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))
