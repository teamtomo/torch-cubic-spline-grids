import pytest
import torch

from torch_cubic_spline_grids import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
    CubicCatmullRomGrid1d,
    CubicCatmullRomGrid2d,
    CubicCatmullRomGrid3d,
    CubicCatmullRomGrid4d,
)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid1d, CubicCatmullRomGrid1d]
)
def test_1d_grid_direct_instantiation(grid_cls):
    """Test grid instantiation with different types for resolution argument."""
    grid = grid_cls()
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (1, 2)

    grid = grid_cls(resolution=5, n_channels=3)
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (3, 5)

    grid = grid_cls(resolution=(5,), n_channels=3)
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (3, 5)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid1d, CubicCatmullRomGrid1d]
)
def test_1d_grid_instantiation_from_existing_data(grid_cls):
    """Test grid instantiation from existing data."""
    grid = grid_cls.from_grid_data(data=torch.zeros(3, 5))
    assert grid.ndim == 1
    assert grid.resolution == (5,)
    assert grid.n_channels == 3
    assert isinstance(grid._data, torch.nn.Parameter)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid1d, CubicCatmullRomGrid1d]
)
def test_calling_1d_grid(grid_cls):
    """Test calling 1d grid."""
    grid = grid_cls()
    expected = torch.tensor([0.])
    for arg in (0.5, [0.5], torch.tensor([0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid1d, CubicCatmullRomGrid1d]
)
def test_1d_grid_with_singleton_dimension(grid_cls):
    """Test that a 2D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = grid_cls(resolution=1)
    result = grid(0.5)
    assert torch.allclose(result, torch.tensor([0.0]))


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid1d, CubicCatmullRomGrid1d]
)
def test_calling_1d_grid_with_stacked_coords(grid_cls):
    """Test calling a 1d grid with a multidimensional array of coordinates."""
    grid = grid_cls(resolution=1)
    h, w = 4, 4

    # no explicit coordinate dimension
    result = grid(torch.rand(size=(h, w)))
    assert result.shape == (h, w)
    assert torch.allclose(result, torch.tensor([0]).float())

    # with explicit coordinate dimension
    result = grid(torch.rand(size=(h, w, 1)))
    assert result.shape == (h, w, 1)
    assert torch.allclose(result, torch.tensor([0]).float())


def test_interpolation_matrix_device():
    """Interpolation matrix should move when Module moves to a different device."""
    grid = CubicBSplineGrid1d(resolution=3)
    assert grid.interpolation_matrix.device == torch.device('cpu')
    grid.to(torch.device('meta'))
    assert grid.interpolation_matrix.device == torch.device('meta')


def test_grid_device():
    """Grid data should move when Module moves to a different device."""
    grid = CubicBSplineGrid1d(resolution=3)
    assert grid.data.device == torch.device('cpu')
    grid.to(torch.device('meta'))
    assert grid.data.device == torch.device('meta')


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid2d, CubicCatmullRomGrid2d]
)
def test_2d_grid_direct_instantiation(grid_cls):
    grid = grid_cls()
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (1, 2, 2)

    grid = grid_cls(resolution=(5, 4), n_channels=3)
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (3, 5, 4)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid2d, CubicCatmullRomGrid2d]
)
def test_2d_grid_instantiation_from_existing_data(grid_cls):
    """Test grid instantiation from existing data."""
    grid = grid_cls.from_grid_data(data=torch.zeros(3, 5, 4))
    assert grid.ndim == 2
    assert grid.resolution == (5, 4)
    assert grid.n_channels == 3
    assert isinstance(grid._data, torch.nn.Parameter)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid2d, CubicCatmullRomGrid2d]
)
def test_calling_2d_grid(grid_cls):
    """Test calling 2d grid."""
    grid = grid_cls()
    expected = torch.tensor([0., 0.])
    for arg in ([0.5, 0.5], torch.tensor([0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid2d, CubicCatmullRomGrid2d]
)
def test_2d_grid_with_singleton_dimension(grid_cls):
    """Test that a 2D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = grid_cls(resolution=(2, 1))
    result = grid([0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0]))

    # singleton in height dim
    grid = grid_cls(resolution=(1, 2))
    result = grid([0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0]))


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid2d, CubicCatmullRomGrid2d]
)
def test_calling_2d_grid_with_stacked_coordinates(grid_cls):
    """Test calling a 2D grid with stacked coordinates."""
    grid = grid_cls(resolution=(2, 2), n_channels=1)
    result = grid(torch.rand(size=(5, 5, 2)))
    assert result.shape == (5, 5, 1)

    grid = grid_cls(resolution=(2, 2), n_channels=2)
    result = grid(torch.rand(size=(5, 5, 2)))
    assert result.shape == (5, 5, 2)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid3d, CubicCatmullRomGrid3d]
)
def test_3d_grid_direct_instantiation(grid_cls):
    grid = grid_cls()
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (1, 2, 2, 2)

    grid = grid_cls(resolution=(5, 4, 3), n_channels=2)
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (2, 5, 4, 3)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid3d, CubicCatmullRomGrid3d]
)
def test_3d_grid_instantiation_from_existing_data(grid_cls):
    """Test grid instantiation from existing data."""
    grid = grid_cls.from_grid_data(data=torch.zeros(2, 5, 4, 3))
    assert grid.ndim == 3
    assert grid.resolution == (5, 4, 3)
    assert grid.n_channels == 2
    assert isinstance(grid._data, torch.nn.Parameter)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid3d, CubicCatmullRomGrid3d]
)
def test_calling_3d_grid(grid_cls):
    """Test calling 3d grid."""
    grid = grid_cls()
    expected = torch.tensor([0., 0., 0.])
    for arg in ([0.5, 0.5, 0.5], torch.tensor([0.5, 0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid3d, CubicCatmullRomGrid3d]
)
def test_calling_3d_grid_with_stacked_coordinates(grid_cls):
    """Test calling 3d grid with stacked coordinates."""
    grid = grid_cls()
    d, h, w = 4, 4, 4
    result = grid(torch.rand(size=(d, h, w, 3)))
    assert result.shape == (d, h, w, 1)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid3d, CubicCatmullRomGrid3d]
)
def test_3d_grid_with_singleton_dimension(grid_cls):
    """Test that a 3D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = grid_cls(resolution=(2, 2, 1))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    # singleton in height dim
    grid = grid_cls(resolution=(2, 1, 2))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))

    # singleton in depth dim
    grid = grid_cls(resolution=(1, 2, 2))
    result = grid([0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0]))


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid4d, CubicCatmullRomGrid4d]
)
def test_4d_grid_direct_instantiation(grid_cls):
    grid = grid_cls()
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (1, 2, 2, 2, 2)

    grid = grid_cls(resolution=(6, 5, 4, 3), n_channels=2)
    assert isinstance(grid, grid_cls)
    assert grid.data.shape == (2, 6, 5, 4, 3)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid4d, CubicCatmullRomGrid4d]
)
def test_4d_grid_instantiation_from_existing_data(grid_cls):
    """Test grid instantiation from existing data."""
    grid = grid_cls.from_grid_data(data=torch.zeros(2, 6, 5, 4, 3))
    assert grid.ndim == 4
    assert grid.resolution == (6, 5, 4, 3)
    assert grid.n_channels == 2
    assert isinstance(grid._data, torch.nn.Parameter)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid4d, CubicCatmullRomGrid4d]
)
def test_calling_4d_grid(grid_cls):
    """Test calling 4d grid."""
    grid = grid_cls()
    expected = torch.tensor([0., 0., 0., 0.])
    for arg in ([0.5, 0.5, 0.5, 0.5], torch.tensor([0.5, 0.5, 0.5, 0.5])):
        result = grid(arg)
        assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid4d, CubicCatmullRomGrid4d]
)
def test_calling_4d_grid_with_stacked_coordinates(grid_cls):
    """Test calling 3d grid with stacked coordinates."""
    grid = grid_cls()
    t, d, h, w = 2, 4, 4, 4
    result = grid(torch.rand(size=(t, d, h, w, 4)))
    assert result.shape == (t, d, h, w, 1)


@pytest.mark.parametrize(
    'grid_cls', [CubicBSplineGrid4d, CubicCatmullRomGrid4d]
)
def test_4d_grid_with_singleton_dimension(grid_cls):
    """Test that a 4D grid with a singleton dimension can be used."""
    # singleton in width dim
    grid = grid_cls(resolution=(2, 2, 2, 1))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in height dim
    grid = grid_cls(resolution=(2, 2, 1, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in depth dim
    grid = grid_cls(resolution=(2, 1, 2, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # singleton in time dim
    grid = grid_cls(resolution=(1, 2, 2, 2))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))

    # multiple singletons
    grid = grid_cls(resolution=(1, 1, 1, 1))
    result = grid([0.5, 0.5, 0.5, 0.5])
    assert torch.allclose(result, torch.tensor([0.0, 0.0, 0.0, 0.0]))
