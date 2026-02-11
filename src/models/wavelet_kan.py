"""
Wavelet KAN (Wav-KAN) for TEP fault detection.

Based on: Bozorgasl & Chen, "Wav-KAN: Wavelet Kolmogorov-Arnold Networks" (2024)
https://arxiv.org/abs/2405.12832

Original code adapted from https://github.com/zavareh1/Wav-KAN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WaveletKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))

        self.weight1 = nn.Parameter(torch.Tensor(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        self.base_activation = nn.SiLU()
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / (scale_expanded + 1e-7)

        if self.wavelet_type == 'mexican_hat':
            term1 = ((x_scaled ** 2) - 1)
            term2 = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = (2 / (math.sqrt(3) * math.pi**0.25)) * term1 * term2
        elif self.wavelet_type == 'morlet':
            omega0 = 5.0
            real = torch.cos(omega0 * x_scaled)
            envelope = torch.exp(-0.5 * x_scaled ** 2)
            wavelet = envelope * real
        elif self.wavelet_type == 'dog':
            wavelet = -x_scaled * torch.exp(-0.5 * x_scaled ** 2)
        elif self.wavelet_type == 'meyer':
            v = torch.abs(x_scaled)
            pi = math.pi

            def nu(t):
                return t**4 * (35 - 84*t + 70*t**2 - 20*t**3)

            def meyer_aux(v):
                return torch.where(
                    v <= 1/2, torch.ones_like(v),
                    torch.where(v >= 1, torch.zeros_like(v),
                                torch.cos(pi / 2 * nu(2 * v - 1))))

            wavelet = torch.sin(pi * v) * meyer_aux(v)
        elif self.wavelet_type == 'shannon':
            pi = math.pi
            sinc = torch.sinc(x_scaled / pi)
            window = torch.hamming_window(
                x_scaled.size(-1), periodic=False,
                dtype=x_scaled.dtype, device=x_scaled.device)
            wavelet = sinc * window
        else:
            raise ValueError(f"Unsupported wavelet type: {self.wavelet_type}")

        wavelet_weighted = wavelet * self.wavelet_weights.unsqueeze(0).expand_as(wavelet)
        wavelet_output = wavelet_weighted.sum(dim=2)
        return wavelet_output

    def forward(self, x):
        wavelet_output = self.wavelet_transform(x)
        combined_output = wavelet_output
        return self.bn(combined_output)


class WaveletKAN(nn.Module):
    """
    Wavelet KAN with standard interface for TEP tuning.

    Parameters
    ----------
    input_dim     : int   — flattened window size (195)
    hidden_dim    : int   — width of hidden layers
    hidden_layers : int   — number of hidden layers
    output_dim    : int   — number of classes (28)
    wavelet_type  : str   — wavelet basis function
    """

    def __init__(self, input_dim: int, hidden_dim: int,
                 hidden_layers: int, output_dim: int,
                 wavelet_type: str = 'mexican_hat'):
        super(WaveletKAN, self).__init__()

        # Build layers_hidden list: [input_dim, hidden_dim, ..., hidden_dim, output_dim]
        dims = [input_dim] + [hidden_dim] * hidden_layers + [output_dim]

        self.layers = nn.ModuleList()
        for in_f, out_f in zip(dims[:-1], dims[1:]):
            self.layers.append(WaveletKANLinear(in_f, out_f, wavelet_type))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
