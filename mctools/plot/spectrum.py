from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt


__all__ = [
    'lorentizian',
    'get_peaks_broadening',

    'BroadeningFunc',
]


BroadeningFunc = Callable[[npt.ArrayLike, float, float], npt.ArrayLike]


def lorentizian(x: npt.ArrayLike, x0: float, fwhm: float) -> np.ndarray:
    x: np.ndarray = np.asarray(x).reshape(-1)
    y: np.ndarray = 2 * (x - x0) / fwhm
    return 1 / (1 + y ** 2)


def get_peaks_broadening(x_position: npt.ArrayLike, y_height: npt.ArrayLike,
                         broadening_func: BroadeningFunc = lorentizian,
                         x_range: tuple[float, float] | None = None, resolution: float | None = None,
                         fwhm: float = 0.1) -> (np.ndarray, np.ndarray):
    # Flatten arrays
    x_position: np.ndarray = np.asarray(x_position, dtype=np.float_).reshape(-1)
    y_height: np.ndarray = np.asarray(y_height, dtype=np.float_).reshape(-1)

    if x_position.shape != y_height.shape:
        raise ValueError(f"Shapes of position and height of peaks must be equal: "
                         f"{x_position.shape} != {y_height.shape}")

    x_min, x_max = x_range if x_range else (x_position.min() - fwhm, x_position.max() + fwhm)
    resolution = resolution if resolution else fwhm / 10

    x = np.arange(x_min, x_max, resolution)
    y = np.zeros_like(x)

    for x0, h in zip(x_position, y_height):
        y += h * broadening_func(x, x0, fwhm)

    return x, y
