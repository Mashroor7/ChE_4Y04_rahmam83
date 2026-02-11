"""KAN model variants for TEP fault detection."""

from .efficient_kan import EfficientKAN
from .fourier_kan import FourierKAN
from .wavelet_kan import WaveletKAN
from .fast_kan import FastKAN

MODEL_REGISTRY = {
    'efficient_kan': EfficientKAN,
    'fourier_kan':   FourierKAN,
    'wavelet_kan':   WaveletKAN,
    'fast_kan':      FastKAN,
}