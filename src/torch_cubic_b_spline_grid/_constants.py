import torch

# Described in Freya Holmer's video "The Continuity of Splines"
# https://youtu.be/jvPPXbo87ds?t=3462

CUBIC_B_SPLINE_MATRIX = (1 / 6) * torch.tensor([[1, 4, 1, 0],
                                                [-3, 0, 3, 0],
                                                [3, -6, 3, 0],
                                                [-1, 3, -3, 1]])

CUBIC_CATMULL_ROM_MATRIX = (1 / 2) * torch.tensor([[0, 2, 0, 0],
                                                   [-1, 0, 1, 0],
                                                   [2, -5, 4, -1],
                                                   [-1, 3, -3, 1]])
