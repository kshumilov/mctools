from __future__ import annotations

from inspect import signature
from typing import Callable, Any

import numpy as np
import numpy.typing as npt

import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks


from ..core import MCTransitions


__all__ = [
    'lorentizian',
    'gaussian',
    'sigmoid',

    'broaden_transitions',

    'BroadeningFunc',
]


EnergyWindow = tuple[float, float] | float | int
BroadeningFunc = Callable[[npt.ArrayLike, float, float, float], npt.ArrayLike]


def gaussian(x: npt.ArrayLike, /, x0: float = 0.0, area: float = 1.0, fwhm: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    x = (x - x0) / sigma
    norm_const = area / (sigma * np.sqrt(2 * np.pi))
    return norm_const * np.exp(-(x ** 2) / 2)


def lorentizian_height_to_area(height: float, fwhm: float = 1.0) -> float:
    return height * fwhm * np.pi / 2


def lorentizian(x: npt.ArrayLike, /, x0: float = 0.0, area: float = 1.0, fwhm: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x)

    x = 2 * (x - x0) / fwhm
    norm_const = area * (2 / (np.pi * fwhm))
    return norm_const / (x ** 2 + 1)


def sigmoid(x: npt.ArrayLike, /, x0: float = 0.0, alpha: float = 1.0, height: float = 1.0) -> npt.NDArray:
    x: npt.NDArray = np.asarray(x).reshape(-1)
    y: npt.NDArray = -alpha * (x - x0)
    return height / (1 + np.exp(y))


def construct_spectrum_fit(funcs: dict[callable, int], constraints: dict[callable, float] = None,
                           funcname: str = 'f', varname: str = 'X'):
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


def estimate_energy_window(
        transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
        fwhm: float = 1.0, n_fwhm: float = 8.
) -> tuple[float, float]:
    """Estimate the energy range in which to construct the broadened spectrum.

    Assuming that each transition's peak area is broadened

    Args:
        transition_energy:
        oscillator_strength:
        fwhm:
        n_fwhm:

    Returns:
        Tuple with lower and upper energy bounds.

    """
    transition_energy = np.asarray(transition_energy, dtype=np.float_)
    oscillator_strength = np.asarray(oscillator_strength, dtype=np.float_)

    if transition_energy.shape != oscillator_strength.shape:
        raise ValueError(f"'transition_energy' and 'oscillator_strength' arrays must have the same shape: "
                         f"{transition_energy.shape} != {oscillator_strength.shape}")

    total_oscillator_strength = oscillator_strength.sum()
    energy_min = np.dot((transition_energy - n_fwhm * fwhm), oscillator_strength) / total_oscillator_strength
    energy_max = np.dot((transition_energy + n_fwhm * fwhm), oscillator_strength) / total_oscillator_strength
    return energy_min, energy_max


def get_energy_window(
        transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
        energy_window: EnergyWindow = 8, fwhm: float = 1.0
) -> tuple(float, float):
    transition_energy = np.asarray(transition_energy, dtype=np.float_)
    oscillator_strength = np.asarray(oscillator_strength, dtype=np.float_)

    match energy_window:
        case (energy_min, energy_max):
            pass
        case (n_fwhm,) | int(n_fwhm) | float(n_fwhm):
            energy_min, energy_max = estimate_energy_window(
                transition_energy, oscillator_strength,
                fwhm=fwhm, n_fwhm=n_fwhm
            )
        case _:
            raise ValueError(f"Invalid 'energy_window': {energy_window}")

    return energy_min, energy_max


def broaden_transitions(
        transition_energy: npt.ArrayLike, oscillator_strength: npt.ArrayLike, /,
        fwhm: npt.ArrayLike | float = 1.0, resolution: float = 1e-3,
        normalize: bool = False,
        energy_window: EnergyWindow = 8, shape_func: callable = lorentizian
) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], float, tuple[float, float]]:
    transition_energy = np.asarray(transition_energy, dtype=np.float_)
    oscillator_strength = np.asarray(oscillator_strength, dtype=np.float_)

    if transition_energy.shape != oscillator_strength.shape:
        raise ValueError(f"'transition_energy' and 'oscillator_strength' arrays must have the same shape: "
                         f"{transition_energy.shape} != {oscillator_strength.shape}")

    energy_min, energy_max = get_energy_window(
        transition_energy, oscillator_strength,
        energy_window=energy_window, fwhm=fwhm
    )
    energy = np.arange(energy_min, energy_max, resolution)
    intensity = np.zeros_like(energy)
    for x0, area in zip(transition_energy, oscillator_strength):
        intensity += shape_func(energy, x0=x0, fwhm=fwhm, area=area)

    total_oscillator_strength = oscillator_strength.sum()
    if normalize:
        intensity /= total_oscillator_strength
        oscillator_strength /= total_oscillator_strength

    return energy, intensity, total_oscillator_strength, (energy_min, energy_max)


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

        guess.extend((x0, area, fwhm))
        lower_bound.extend((xl, .0, .0))
        upper_bound.extend((xr, np.inf, 10.,))

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

    param_cols = [center_col, area_col, fwhm_col]
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
        energy_window: EnergyWindow = 8,
        peak_height: float = 0.01,
        peak_energy_col: str = 'E0', peak_intensity_col: str = 'I0',
        area_penalty_fit: float = 0.0
):
    energy, intensity, norm, *_ = broaden_transitions(
        transition_energy, oscillator_strength, normalize=normalize,
        fwhm=fwhm, resolution=resolution, energy_window=energy_window, shape_func=lorentizian
    )

    peak_idx = find_peaks_savgol(intensity, resolution=resolution, peak_height=peak_height)
    peak_energy, peak_intensity = energy[peak_idx], intensity[peak_idx]
    guess, *bounds = form_lorentizian_spectrum_guess(peak_energy, peak_intensity, fwhm=fwhm)

    n_peaks = len(peak_idx)
    if area_penalty_fit != 0.:
        penalty = construct_area_penalty(n_peaks, area=np.trapz(intensity, energy))
        fitting_func = construct_spectrum_fit({lorentizian: n_peaks}, constraints={penalty: area_penalty_fit})
    else:
        fitting_func = construct_spectrum_fit({lorentizian: n_peaks})

    peaks = fit_lorentizian_spectrum(energy, intensity, fitting_func, guess=guess, bounds=bounds)
    peaks[peak_energy_col] = peak_energy
    peaks[peak_intensity_col] = peak_intensity
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


def plot_spectrum(
        transition_energy: npt.ArrayLike,
        oscillator_strength: npt.ArrayLike,
        /,
        ax: plt.Axes,
        *,
        # Plotting parameters
        curve_style: dict = None,
        fill_style: dict = None,
        sticks_style: dict = None,

        # Shape of the spectrum
        energy_shift: float = 0.0,
        intensity_scaler: float = 1.0,
        intensity_shift: float = 0.0,

        # Broadening Parameters
        fwhm: float = 1.0,
        resolution: float = 0.1,
        energy_window: EnergyWindow = 10,
        shape_func: callable = lorentizian,
        normalize: bool = True,
) -> dict[str, Any]:
    transition_energy = np.asarray(transition_energy, dtype=np.float64)
    oscillator_strength = np.asarray(oscillator_strength, dtype=np.float64)

    if transition_energy.shape != oscillator_strength.shape:
        raise ValueError(f"'transition_energy' and 'oscillator_strength' arrays must have the same shape: "
                         f"{transition_energy.shape} != {oscillator_strength.shape}")

    data = {
        'transition_energy': transition_energy,
        'oscillator_strength': oscillator_strength
    }

    transition_energy += energy_shift

    energy_min, energy_max = data.setdefault(
        'energy_window',
        get_energy_window(
            transition_energy, oscillator_strength,
            energy_window=energy_window, fwhm=fwhm
        )
    )

    total_oscillator_strength = data.setdefault('total_oscillator_strength', oscillator_strength.sum())
    if normalize:
        oscillator_strength /= total_oscillator_strength

    oscillator_strength *= intensity_scaler

    if sticks_style is not None:
        window = (energy_min <= transition_energy) & (transition_energy <= energy_max)

        curr_ax = sticks_style.pop('ax', ax)
        curr_ax.vlines(
            transition_energy[window], intensity_shift, intensity_shift + oscillator_strength[window],
            **sticks_style
        )

    if curve_style is not None or fill_style is not None:
        energy, intensity, *_ = broaden_transitions(
            transition_energy, oscillator_strength,
            normalize=False,
            energy_window=(energy_min, energy_max),
            fwhm=fwhm, resolution=resolution, shape_func=shape_func,
        )

        intensity += intensity_shift

        if curve_style is not None:
            curr_ax = curve_style.pop('ax', ax)
            curr_ax.plot(energy, intensity + intensity_shift, **curve_style)

        if fill_style is not None:
            curr_ax = fill_style.pop('ax', ax)
            curr_ax.fill_between(energy, intensity + intensity_shift, **fill_style)

        data['intensity'] = intensity
        data['energy'] = energy

    return data


def plot_spectra(
        transitions: pd.DataFrame,
        /,
        ax: plt.Axes,
        *,
        # Plotting the spectrum main feature
        curve_style: dict = None,
        fill_style: dict = None,
        sticks_style: dict = None,

        # Dict of conditions to select subspectra
        subspectra_cond: dict[tuple[str, callable], dict] = None,

        # Linear position and shape of the spectrum
        energy_shift: float = 0.0,
        intensity_scaler: float = 1.0,
        intensity_shift: float = 0.0,

        # Broadening Parameters
        bright_threshold: float = 0.0,
        fwhm: float = 1.0,
        resolution: float = 0.1,
        energy_window: EnergyWindow = 10,
        shape_func: callable = lorentizian,
        normalize: bool = True,

        # Data parameters
        transition_energy_col: str = MCTransitions.RELATIVE_DE_COL,
        oscillator_strength_col: str = MCTransitions.OSC_COL,
):
    # Filter out dark excitations with oscillator_strength below threshold
    transitions = transitions[transitions[oscillator_strength_col] >= bright_threshold]
    transition_energy, oscillator_strength = transitions[[transition_energy_col, oscillator_strength_col]].T.values.copy()

    spectra_data = plot_spectrum(
        transition_energy, oscillator_strength,
        ax,

        # Plotting the spectrum main feature
        curve_style=curve_style,
        fill_style=fill_style,
        sticks_style=sticks_style,

        # Linear position and shape of the spectrum
        energy_shift=energy_shift,
        intensity_scaler=intensity_scaler,
        intensity_shift=intensity_shift,

        # Broadening Parameters
        fwhm=fwhm, resolution=resolution,
        energy_window=energy_window,
        normalize=normalize, shape_func=shape_func,
    )

    transition_energy = spectra_data.get('transition_energy', transition_energy)
    oscillator_strength = spectra_data.get('oscillator_strength', oscillator_strength)
    total_oscillator_strength = spectra_data.get('total_oscillator_strength')

    subspectra_data = spectra_data.setdefault('subspectra', dict())
    subspectra_cond = subspectra_cond if subspectra_cond is not None else {}
    for (sub_label, sub_cond), sub_style in subspectra_cond.items():
        selection = sub_cond(transitions)
        subspectra_data[sub_label] = plot_spectrum(
            transition_energy[selection], oscillator_strength[selection],
            ax,

            # Plotting the spectrum main feature
            **sub_style,

            # Linear position and shape of the spectrum
            energy_shift=0.0,
            intensity_scaler=1.0,
            intensity_shift=0.0,

            # Broadening Parameters
            fwhm=fwhm, resolution=resolution,
            energy_window=spectra_data.get('energy_window', energy_window),
            shape_func=shape_func, normalize=False
        )

        subspectra_data[sub_label]['total_oscillator_strength'] *= total_oscillator_strength

    return spectra_data
