from typing import Callable

import numpy as np
import numpy.typing as npt

__all__ = [
    'lorentizian',
    'get_peaks_broadening',

    'BroadeningFunc'
]


BroadeningFunc = Callable[[npt.ArrayLike, float, float], npt.ArrayLike]


def lorentizian(x: npt.ArrayLike, x0: float, fwhm: float) -> np.ndarray:
    x: np.ndarray = np.asarray(x).reshape(-1)
    y: np.ndarray = 2 * (x - x0) / fwhm
    return 1 / (1 + y ** 2)


def get_peaks_broadening(peak: npt.ArrayLike, height: npt.ArrayLike, broadening_func: BroadeningFunc = lorentizian,
                         x_range: tuple[float, float] | None = None, resolution: float | None = None,
                         fwhm: float = 0.1) -> (np.ndarray, np.ndarray):
    peak: np.ndarray = np.asarray(peak, dtype=np.float_).reshape(-1)
    height: np.ndarray = np.asarray(height, dtype=np.float_).reshape(-1)

    assert peak.shape == height.shape

    x_min, x_max = x_range if x_range else (peak.min() - fwhm, peak.max() + fwhm)
    resolution = resolution if resolution else fwhm / 10

    x = np.arange(x_min, x_max, resolution)
    y = np.zeros_like(x)

    for x0, h in zip(peak, height):
        y += h * broadening_func(x, x0, fwhm)

    return x, y
