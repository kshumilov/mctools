from __future__ import annotations

from inspect import signature
from typing import Callable

import numpy as np
import numpy.typing as npt

import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks


__all__ = [
    'lorentizian',
    'gaussian',
    'sigmoid',

    'broaden_transitions',

    'BroadeningFunc',
]


BroadeningFunc = Callable[[npt.ArrayLike, float, float, float], npt.ArrayLike]


def gaussian(x: npt.ArrayLike, /, x0: float = 0.0, fwhm: float = 1.0, area: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = (x - x0) / sigma
    norm_const = area / (sigma * np.sqrt(2 * np.pi))
    return norm_const * np.exp(-(x ** 2) / 2)


def lorentizian_height_to_area(height: float, fwhm: float = 1.0) -> float:
    return height * fwhm * np.pi / 2


def lorentizian(x: npt.ArrayLike, /, x0: float = 0.0, fwhm: float = 1.0, area: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)

    x = 2 * (x - x0) / fwhm
    norm_const = area * (2 / (np.pi * fwhm))
    return norm_const / (x ** 2 + 1)


def sigmoid(x: npt.ArrayLike, /, x0: float = 0.0, alpha: float = 1.0, height: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x).reshape(-1)
    y: npt.NDArray = -alpha * (x - x0)
    return height / (1 + np.exp(y))


# def get_peaks_broadening(position: npt.ArrayLike | str, area: npt.ArrayLike | str,
#                          broadening_func: BroadeningFunc = lorentizian,
#                          x_range: tuple[float, float] | None = None, resolution: float | None = None, fwhm: float = 0.1,
#                          normalize: bool | float = False, scale_y: float = 1.0,
#                          data: Mapping[str, npt.ArrayLike] | None = None) -> (np.ndarray, np.ndarray):
#
#     if isinstance(position, str):
#         position = data[position]
#
#     if isinstance(area, str):
#         area = data[area]
#
#     # Flatten arrays
#     position = np.asarray(position, dtype=np.float_).reshape(-1)
#     area = np.asarray(area, dtype=np.float_).reshape(-1)
#
#     if position.shape != area.shape:
#         raise ValueError(f"Shapes of position and area of peaks must be equal: "
#                          f"{position.shape} != {area.shape}")
#
#     x_min, x_max = x_range if x_range else (position.min() - fwhm, position.max() + fwhm)
#     resolution = resolution if resolution else fwhm / 10
#
#     position = np.arange(x_min, x_max, resolution)
#     y = np.zeros_like(position)
#
#     for x0, A in zip(position, area):
#         y += broadening_func(position, x0, fwhm, A)
#
#     if normalize:
#         if float(normalize) != 1.0:
#             y /= normalize
#         else:
#             y /= y.max()
#
#     return position, y * scale_y


def construct_spectrum_fit(funcs: dict[callable, int], constraints: dict[callable, float] = None, funcname: str = 'f',
                           varname: str = 'X'):
    arglist = []
    funclist = []

    for func, repeats in funcs.items():
        func_params = list(signature(func).parameters)

        n_args = len(func_params) - 1
        func_args_tmplt = ', '.join([f'{p}_%d' for p in func_params[1:]])  # f(X, *params)
        func_sign_tmplt = f'{func.__name__}({varname}, %s)'

        for _ in range(repeats):
            i = len(funclist) + 1
            func_args = func_args_tmplt % ((i,) * n_args)

            arglist.append(func_args)
            funclist.append(func_sign_tmplt % func_args)

    constraints = {} if constraints is None else constraints

    for func, penalty in constraints.items():
        func_params = ', '.join(list(signature(func).parameters))
        funclist.append(f'{func.__name__}({func_params}) * {penalty}')

    args = ', '.join(arglist)
    func_code = ' + '.join(funclist)
    func_code = f"def {funcname}({varname}, {args}):\n    return {func_code}"

    fakeglobals = {}
    compiled_code = compile(func_code, "custom_spectra_fit", "exec")

    funcdefs = {f.__name__: f for f in funcs.keys()}
    funcdefs |= {f.__name__: f for f in constraints.keys()}
    eval(compiled_code, funcdefs, fakeglobals)

    return fakeglobals[funcname]


def construct_area_penalty(n_peaks: int, area: float = 1.0) -> callable:
    arglist = [f'area_{i + 1}' for i in range(n_peaks)]

    args = ', '.join(arglist)
    func_code = '[%s]' % args
    func_code = f"def area_penalty({args}):\n    return abs({area} - sum({func_code}))"

    fakeglobals = {}
    compiled_code = compile(func_code, "custom_penalty", "exec")
    eval(compiled_code, {}, fakeglobals)
    return fakeglobals['area_penalty']


def estimate_energy_window(transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
                           fwhm: float = 1.0, n_fwhm: float = 8.) -> tuple[float, float]:
    transition_energy = np.asarray(transition_energy)
    oscillator_strength = np.asarray(oscillator_strength)

    oscillator_strength_norm = oscillator_strength / oscillator_strength.sum()
    energy_min = np.sum((transition_energy - n_fwhm * fwhm) * oscillator_strength_norm)
    energy_max = np.sum((transition_energy + n_fwhm * fwhm) * oscillator_strength_norm)
    return energy_min, energy_max


def broaden_transitions(
        transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
        fwhm: npt.ArrayLike | float = 1.0, resolution: float = 1e-3, bright_threshold: float = 0.0,
        normalize: bool = False,
        energy_window: tuple[float, float] | float | int = 8, shape_func: callable = lorentizian
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], float]:
    transition_energy = np.asarray(transition_energy)
    oscillator_strength = np.asarray(oscillator_strength)

    if transition_energy.shape != oscillator_strength.shape:
        raise ValueError(f"'transition_energy' and 'oscillator_strength' arrays must have the same shape: "
                         f"{transition_energy.shape} != {oscillator_strength.shape}")

    # Filter out dark excitations with oscillator_strength below threshold
    is_bright = oscillator_strength > bright_threshold
    transition_energy = transition_energy[is_bright]
    oscillator_strength = oscillator_strength[is_bright]

    match energy_window:
        case (energy_min, energy_max):
            pass
        case (n_fwhm, ) | int(n_fwhm) | float(n_fwhm):
            energy_min, energy_max = estimate_energy_window(transition_energy, oscillator_strength,
                                                            fwhm=fwhm, n_fwhm=n_fwhm)
        case _:
            raise ValueError(f"Invalid 'energy_window': {energy_window}")

    energy = np.arange(energy_min, energy_max, resolution)
    intensity = np.zeros_like(energy)
    for x0, area in zip(transition_energy, oscillator_strength):
        intensity += shape_func(energy, x0=x0, fwhm=fwhm, area=area)

    total_oscillator_strength = oscillator_strength.sum()
    if normalize:
        intensity /= total_oscillator_strength

    return energy, intensity, total_oscillator_strength


def form_lorentizian_spectrum_guess(peak_energy: npt.ArrayLike, peak_intensity: npt.ArrayLike, /,
                                    fwhm: float = 1.0, x0_window: float = 0.5):
    peak_energy = np.asarray(peak_energy)
    peak_intensity = np.asarray(peak_intensity)

    if peak_energy.shape != peak_intensity.shape:
        raise ValueError(f'peak_energy and peak_intensity arrays must have the same shape: '
                         f'{peak_energy.shape} != {peak_intensity.shape}')

    guess = []
    lower_bound = []
    upper_bound = []
    for x0, h0 in zip(peak_energy, peak_intensity):
        xl, xr = x0 - x0_window * fwhm, x0 + x0_window * fwhm
        area = h0 * fwhm * np.pi / 2

        guess.extend((x0, fwhm, area))
        lower_bound.extend((xl, .0, .0))
        upper_bound.extend((xr, 10., np.inf))

    return guess, lower_bound, upper_bound


def fit_lorentizian_spectrum(
        energy: npt.ArrayLike, intensity: npt.ArrayLike, f: callable, /,
        guess=None, bounds=None, peak_width: float = 1.0,
        center_col: str = 'x0', fwhm_col: str = 'fwhm', area_col: str = 'area',
        height_col: str = 'h', lower_bound_col: str = 'xl', upper_bound_col: str = 'xr',
):
    energy = np.asarray(energy)
    intensity = np.asarray(intensity)

    popt, pcov = curve_fit(f, energy, intensity, p0=guess, bounds=bounds)
    perr = np.sqrt(np.diag(pcov))

    param_cols = [center_col, fwhm_col, area_col]
    peaks = pd.DataFrame(popt.reshape(-1, len(param_cols)), columns=param_cols)
    param_err_cols = [f'{col}_err' for col in param_cols]
    peaks[param_err_cols] = perr.reshape(-1, len(param_cols))

    peaks[height_col] = 2 * peaks[area_col] / (peaks[fwhm_col] * np.pi)
    peaks[lower_bound_col] = peaks[center_col] - (peak_width / 2) * peaks[fwhm_col]
    peaks[upper_bound_col] = peaks[center_col] + (peak_width / 2) * peaks[fwhm_col]

    peaks.sort_values(center_col, inplace=True)
    return peaks


def find_peaks_savgol(
        intensity: npt.ArrayLike, /,
        resolution: float | None = None, peak_height: float = 0.01
) -> npt.NDArray[np.int_]:
    intensity = np.asarray(intensity)

    d2_intensity = -savgol_filter(intensity, 16, 2, deriv=2, delta=resolution)
    peak_idx, info = find_peaks(d2_intensity, height=peak_height)
    peak_idx = peak_idx[d2_intensity[peak_idx] > 0]
    return peak_idx


def identify_lorentizian_peaks(
        transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
        normalize=True, fwhm: float = 1.0, resolution: float = .1,
        energy_window: tuple[float, float] | float | int = 8, peak_height: float = 0.01,
        peak_energy_col: str = 'E0', peak_intensity_col: str = 'I0', area_penalty_fit: float = 0.0
):
    energy, intensity, norm = broaden_transitions(
        transition_energy, oscillator_strength, normalize=normalize,
        fwhm=fwhm, resolution=resolution, energy_window=energy_window, shape_func=lorentizian
    )

    peak_idx = find_peaks_savgol(intensity, resolution=resolution, peak_height=peak_height)
    n_peaks = len(peak_idx)

    peak_energy, peak_intensity = energy[peak_idx], intensity[peak_idx]
    guess, *bounds = form_lorentizian_spectrum_guess(peak_energy, peak_intensity, fwhm=fwhm)

    if area_penalty_fit != 0.:
        penalty = construct_area_penalty(n_peaks, area=np.trapz(intensity, energy))
        fitting_func = construct_spectrum_fit({lorentizian: n_peaks}, constraints={penalty: area_penalty_fit})
    else:
        fitting_func = construct_spectrum_fit({lorentizian: n_peaks})

    peaks = fit_lorentizian_spectrum(energy, intensity, fitting_func, guess=guess, bounds=bounds)
    peaks[[peak_energy_col, peak_intensity_col]] = peak_energy, peak_intensity
    return peaks, fitting_func, energy, intensity, norm


def get_peak_blocked_rdm_diag(
        peak: pd.Series, df_transitions: pd.DataFrame, mo_blocks: list[str], /,
        transition_energy_col: str = 'dE0', oscillator_strength_col: str = 'osc',
        lower_bound_col: str = 'xl', upper_bound_col: str = 'xr'
):
    # Select transitions in the energy window of the peak (xl, xr)
    peak_window = df_transitions[transition_energy_col].between(peak[lower_bound_col], peak[upper_bound_col])
    peak_transitions = df_transitions[peak_window]

    # Calculate transitions' contribution to the peak by mo_block
    oscillator_strength = peak_transitions[oscillator_strength_col]
    weight = oscillator_strength / oscillator_strength.sum()
    rdm_diag_mean = peak_transitions[mo_blocks].mul(weight, axis=0).sum(axis=0)
    return rdm_diag_mean


def label_peaks_new(
        peaks: pd.DataFrame, df_transitions: pd.DataFrame, mo_blocks: list[str], /,
        labels: dict[str, callable] | None = None, default_label: str = '',
        transition_energy_col: str = 'dE0', oscillator_strength_col: str = 'osc',
        center_col: str = 'x0', lower_bound_col: str = 'xl', upper_bound_col: str = 'xr',
        label_col: str = 'label'
):
    rdm_diag_mean = peaks.apply(get_peak_blocked_rdm_diag, axis=1, args=(df_transitions, mo_blocks),
                                transition_energy_col=transition_energy_col,
                                oscillator_strength_col=oscillator_strength_col,
                                lower_bound_col=lower_bound_col, uppwer_bound_col=upper_bound_col)
    peaks = pd.concat((peaks, rdm_diag_mean), axis=1, copy=False)

    peaks[label_col] = default_label

    labels = labels if labels is not None else {}
    for label, cond in labels.items():
        idx = peaks[cond(peaks)].sort_values(center_col).index
        peaks.loc[idx, label_col] = [f'{label}{i}' for i in range(1, len(idx) + 1)]

    return peaks
