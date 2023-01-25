import torch

from torch_cubic_b_spline_grid import CubicBSplineGrid1d


def test_1d_grid_optimisation():
    n_control_points = 6
    n_observations_per_iteration = 20
    grid = CubicBSplineGrid1d(resolution=n_control_points, n_channels=1)

    def f(x: torch.Tensor, add_noise: bool = False):
        y = torch.sin(x * 2 * torch.pi)
        if add_noise is True:
            y += torch.normal(mean=torch.zeros(len(y)), std=0.3)
        return y

    optimiser = torch.optim.SGD(lr=0.02, params=grid.parameters())
    for i in range(5000):
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

