from torch_cubic_spline_grids import (
    CubicBSplineGrid1d,
    CubicBSplineGrid2d,
    CubicBSplineGrid3d,
    CubicBSplineGrid4d,
)


def test_grid_class_instantiation():
    grid_classes = [
        CubicBSplineGrid1d,
        CubicBSplineGrid2d,
        CubicBSplineGrid3d,
        CubicBSplineGrid4d
    ]
    for grid_class in grid_classes:
        instance = grid_class()
        assert isinstance(instance, grid_class)
        assert len(list(instance.parameters())) > 0