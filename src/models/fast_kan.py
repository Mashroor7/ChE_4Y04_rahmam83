"""
Fast KAN for TEP fault detection.

Uses Radial Basis Functions (RBFs) instead of B-splines for faster computation.

Original code adapted from https://github.com/ZiyaoLi/fast-kan
Licensed under Apache 2.0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int,
                 init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(self, grid_min: float = -2., grid_max: float = 2.,
                 num_grids: int = 8, denominator: float = None):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int,
                 grid_min: float = -2., grid_max: float = 2.,
                 num_grids: int = 8, use_base_update: bool = True,
                 use_layernorm: bool = True, base_activation=F.silu,
                 spline_weight_init_scale: float = 0.1) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm and input_dim > 1:
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, use_layernorm=True):
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    """
    Fast KAN with standard interface for TEP tuning.

    Parameters
    ----------
    input_dim  : int   — flattened window size (195)
    hidden_dim : int   — width of hidden layers
    hidden_layers : int — number of hidden layers
    output_dim : int   — number of classes (28)
    num_grids  : int   — number of RBF grid points
    grid_min   : float — lower bound of RBF grid
    grid_max   : float — upper bound of RBF grid
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 hidden_layers: int, output_dim: int,
                 num_grids: int = 8, grid_min: float = -2.,
                 grid_max: float = 2.):
        super(FastKAN, self).__init__()

        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=True,
                base_activation=F.silu,
                spline_weight_init_scale=0.1,
            )
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
