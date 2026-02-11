"""
Efficient KAN for TEP fault detection.

Uses B-splines as the basis function for learned activations.

Original code adapted from https://github.com/Blealtan/efficient-kan
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5,
                 spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True,
                 base_activation=nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(EfficientKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h
             + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                 - 1 / 2)
                * self.scale_noise / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline
                 if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order:-self.spline_order], noise)
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, :-(k + 1)])
                / (grid[:, k:-1] - grid[:, :-(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1:] - x)
                / (grid[:, k + 1:] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )
        assert bases.size() == (
            x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        result = solution.permute(2, 0, 1)
        assert result.size() == (
            self.out_features, self.in_features,
            self.grid_size + self.spline_order)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.out_features)
        return output


class EfficientKAN(nn.Module):
    """
    Efficient KAN (B-spline) with standard interface for TEP tuning.

    Parameters
    ----------
    input_dim     : int — flattened window size (195)
    hidden_dim    : int — width of hidden layers
    hidden_layers : int — number of hidden layers
    output_dim    : int — number of classes (28)
    grid_size     : int — number of B-spline grid intervals
    spline_order  : int — degree of the B-spline
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 hidden_layers: int, output_dim: int,
                 grid_size: int = 5, spline_order: int = 3):
        super(EfficientKAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

        self.layers = nn.ModuleList()
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            self.layers.append(
                EfficientKANLinear(
                    in_f, out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
