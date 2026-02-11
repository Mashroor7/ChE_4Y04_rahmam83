"""
Fourier KAN for TEP fault detection.

Uses 1D Fourier coefficients instead of splines for the learned activation functions.
Fourier basis is global (vs local splines), periodic, and numerically bounded.

Original code adapted from https://github.com/GistNoesis/FourierKAN
"""

import torch
import torch.nn as nn
import numpy as np


class FourierKANLinear(nn.Module):
    def __init__(self, inputdim, outdim, gridsize, addbias=True,
                 smooth_initialization=False):
        super(FourierKANLinear, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        grid_norm_factor = (
            (torch.arange(gridsize) + 1)**2
            if smooth_initialization
            else np.sqrt(gridsize)
        )

        self.fouriercoeffs = nn.Parameter(
            torch.randn(2, outdim, inputdim, gridsize)
            / (np.sqrt(inputdim) * grid_norm_factor)
        )
        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        xshp = x.shape
        outshape = xshp[0:-1] + (self.outdim,)
        x = torch.reshape(x, (-1, self.inputdim))

        k = torch.reshape(
            torch.arange(1, self.gridsize + 1, device=x.device),
            (1, 1, 1, self.gridsize)
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))

        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        y = torch.sum(c * self.fouriercoeffs[0:1], (-2, -1))
        y += torch.sum(s * self.fouriercoeffs[1:2], (-2, -1))
        if self.addbias:
            y += self.bias

        y = torch.reshape(y, outshape)
        return y


class FourierKAN(nn.Module):
    """
    Fourier KAN with standard interface for TEP tuning.

    Parameters
    ----------
    input_dim             : int  — flattened window size (195)
    hidden_dim            : int  — width of hidden layers
    hidden_layers         : int  — number of hidden layers
    output_dim            : int  — number of classes (28)
    gridsize              : int  — number of Fourier frequencies per dimension
    smooth_initialization : bool — attenuate high-frequency init
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 hidden_layers: int, output_dim: int,
                 gridsize: int = 8, smooth_initialization: bool = False):
        super(FourierKAN, self).__init__()

        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

        self.layers = nn.ModuleList([
            FourierKANLinear(
                in_dim, out_dim,
                gridsize=gridsize,
                addbias=True,
                smooth_initialization=smooth_initialization,
            )
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
