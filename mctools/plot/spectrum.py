from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import numpy.typing as npt


__all__ = [
    'lorentizian',
    'gaussian',
    'sigmoid',

    'get_peaks_broadening',

    'BroadeningFunc',
]

import pandas as pd

BroadeningFunc = Callable[[npt.ArrayLike, float, float, float], npt.ArrayLike]


def gaussian(x: npt.ArrayLike, /, x0: float = 0.0, fwhm: float = 1.0, height: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x).reshape(-1)
    y: npt.NDArray = -(x - x0) / fwhm
    return height / (fwhm * np.sqrt(2 * np.pi)) * np.exp(-(y ** 2) / 2)


def lorentizian(x: npt.ArrayLike, /, x0: float = 0.0, fwhm: float = 1.0, height: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x).reshape(-1)
    y: npt.NDArray = 2 * (x - x0) / fwhm
    return height / (1 + y ** 2)


def sigmoid(x: npt.ArrayLike, /, x0: float = 0.0, fwhm: float = 1.0, height: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x).reshape(-1)
    y: npt.NDArray = -fwhm * (x - x0)
    return height / (1 + np.exp(y))


def get_peaks_broadening(x_position: npt.ArrayLike | str, y_height: npt.ArrayLike | str,
                         broadening_func: BroadeningFunc = lorentizian,
                         x_range: tuple[float, float] | None = None, resolution: float | None = None, fwhm: float = 0.1,
                         normalize: bool | float = False, scale_y: float = 1.0,
                         data: Mapping[str, npt.ArrayLike] | None = None) -> (np.ndarray, np.ndarray):

    if isinstance(x_position, str):
        x_position = data[x_position]

    if isinstance(y_height, str):
        y_height = data[y_height]

    # Flatten arrays
    x_position = np.asarray(x_position, dtype=np.float_).reshape(-1)
    y_height = np.asarray(y_height, dtype=np.float_).reshape(-1)

    if x_position.shape != y_height.shape:
        raise ValueError(f"Shapes of position and height of peaks must be equal: "
                         f"{x_position.shape} != {y_height.shape}")

    x_min, x_max = x_range if x_range else (x_position.min() - fwhm, x_position.max() + fwhm)
    resolution = resolution if resolution else fwhm / 10

    x = np.arange(x_min, x_max, resolution)
    y = np.zeros_like(x)

    for x0, h in zip(x_position, y_height):
        y += broadening_func(x, x0, fwhm, h)

    if normalize:
        if float(normalize) != 1.0:
            y /= normalize
        else:
            y /= y.max()

    return x, y * scale_y
