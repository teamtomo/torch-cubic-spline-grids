import einops
import torch

from torch_cubic_b_spline_grid import CubicBSplineGrid1d, CubicBSplineGrid2d, \
    CubicBSplineGrid3d


def test_1d_grid_optimisation():
    grid_resolution = 6
    n_observations_per_iteration = 20
    grid = CubicBSplineGrid1d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor, add_noise: bool = False):
        y = torch.sin(x * 2 * torch.pi)
        if add_noise is True:
            y += torch.normal(mean=torch.zeros(len(y)), std=0.3)
        return y

    optimiser = torch.optim.SGD(lr=0.1, params=grid.parameters())
    for i in range(1000):
        x = torch.rand(size=(n_observations_per_iteration, ))
        observations = f(x, add_noise=True)
        prediction = grid(x).squeeze()
        loss = torch.mean(torch.abs(prediction - observations))
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    x = torch.linspace(0, 1, steps=100)
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02


def test_2d_grid_optimisation():
    grid_resolution = (6, 6)
    n_observations_per_iteration = 1000
    grid = CubicBSplineGrid2d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor, add_noise: bool = False):
        centered = x - 0.5
        y = torch.sqrt(torch.sum(centered**2, dim=-1))  # (x**2 + y**2) ** 0.5
        if add_noise is True:
            y += torch.normal(mean=torch.zeros(len(y)), std=0.3)
        return y

    optimiser = torch.optim.SGD(lr=0.1, params=grid.parameters())
    for i in range(1000):
        x = torch.rand(size=(n_observations_per_iteration, 2))
        observations = f(x, add_noise=True)
        prediction = grid(x).squeeze()
        loss = torch.mean((prediction - observations)**2)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    _x = torch.linspace(0, 1, steps=100)
    x = torch.meshgrid(_x, _x, indexing='xy')
    x = einops.rearrange([*x], 'xy h w -> (h w) xy')
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02


def test_3d_grid_optimisation():
    grid_resolution = (6, 6, 6)
    n_observations_per_iteration = 1000
    grid = CubicBSplineGrid3d(resolution=grid_resolution, n_channels=1)

    def f(x: torch.Tensor, add_noise: bool = False):
        centered = x - 0.5
        y = torch.sqrt(torch.sum(centered**2, dim=-1))  # (x**2 + y**2 + z**2) ** 0.5
        if add_noise is True:
            y += torch.normal(mean=torch.zeros(len(y)), std=0.3)
        return y

    optimiser = torch.optim.SGD(lr=1, params=grid.parameters())
    for i in range(1000):
        x = torch.rand(size=(n_observations_per_iteration, 3))
        observations = f(x, add_noise=True)
        prediction = grid(x).squeeze()
        loss = torch.mean((prediction - observations)**2)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

    _x = torch.linspace(0, 1, steps=100)
    x = torch.meshgrid(_x, _x, _x, indexing='xy')
    x = einops.rearrange([*x], 'xyz d h w -> (d h w) xyz')
    ground_truth = f(x)
    prediction = grid(x).squeeze()
    mean_absolute_error = torch.mean(torch.abs(prediction - ground_truth))
    assert mean_absolute_error.item() < 0.02

