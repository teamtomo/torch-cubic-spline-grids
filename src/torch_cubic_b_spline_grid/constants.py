import torch

# The Continuity of Splines: https://youtu.be/jvPPXbo87ds?t=3224
CUBIC_B_SPLINE_MATRIX = (1 / 6) * torch.tensor([[1, 4, 1, 0],
                                                [-3, 0, 3, 0],
                                                [3, -6, 3, 0],
                                                [-1, 3, -3, 1]])
