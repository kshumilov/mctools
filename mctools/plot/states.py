from typing import Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from ..core import MCStates

__all__ = [
    'prepare_state_centers',
    'plot_state_levels',
]


def prepare_state_centers(energy: npt.ArrayLike, degeneracy: npt.ArrayLike, /,
                          column_center: float = 0., *,
                          max_col_width: float = 1.) -> np.ndarray:
    energy = np.asarray(energy)
    degeneracy = np.asarray(degeneracy)

    if energy.shape != degeneracy.shape:
        raise ValueError(
            f"'energy' and 'degeneracy' arrays must have the same shape: {energy.shape} != {degeneracy.shape}")

    if energy.ndim != 1:
        raise ValueError(f"'energy' and 'degeneracy' arrays must be one dimensional: {energy.ndim} != 1")

    state_center = np.zeros_like(energy) + (column_center - max_col_width / 2)
    for d, n in zip(*np.unique(degeneracy, return_counts=True)):
        l = max_col_width / n
        j = 2 * np.arange(n) + 1
        state_center[(degeneracy == d)] += j * l / 2
    return state_center


def plot_state_levels(
        ax: plt.Axes,

        data: MCStates | Sequence[MCStates],
        energy_col: str = MCStates.RELATIVE_ENERGY_COL,
        degeneracy_col: str = MCStates.DEGENERACY_COL,
        state_label_col: str = MCStates.DOMINANT_CONFIG_COL,

        state_labels: dict[str, str] | None = None,
        xlabels: Sequence[str] | None = None,

        title: str | None = None,
        xlabel: str = 'Structure',
        ylabel: str = r'$E-E_0$',
        legend_title: str = 'States',

        center_scale: float = 1.,
        column_width: float = 1.,

        level_width: float = .05,
        level_thickness: float = 3.,
        label_size: int = 18,
):
    if isinstance(data, MCStates):
        data = [data]

    state_labels = state_labels if state_labels else {}

    xticks = []
    for center_idx, states in enumerate(data):
        center_idx *= center_scale
        state_centers = prepare_state_centers(states.df[energy_col], states.df[degeneracy_col],
                                              column_center=center_idx,
                                              max_col_width=column_width)
        xticks.append(center_idx)

        if MCStates.DOMINANT_CONFIG_COL in states.property_columns:
            iterator = states.df.groupby(state_label_col)
        else:
            iterator = zip([None], [states.df])

        for i, (label, df) in enumerate(iterator):
            E = df[energy_col]
            state_end = state_centers[df.index.values] + (level_width / 2)
            state_start = state_centers[df.index.values] - (level_width / 2)

            label = state_labels.get(label, label) if center_idx == 0 else None
            ax.hlines(E, state_start, state_end, label=label,
                      color=f'C{i}', lw=level_thickness)

    xticks = np.asarray(xticks)
    for x in (xticks[1:] + xticks[:-1]) / 2:
        ax.axvline(x, c='k', ls='dotted', alpha=0.2)

    ax.spines['bottom'].set_visible(False)
    ax.spines['bottom'].set_position(('outward', 10))

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if xlabels:
        ax.set_xticks(xticks, list(xlabels))
    else:
        ax.set_xticks(xticks, xticks)

    ax.tick_params(axis='both', labelsize=16)

    if title:
        ax.set_title(title, size=label_size)

    if ylabel:
        ax.set_ylabel(ylabel, size=label_size)

    if xlabel:
        ax.set_xlabel(xlabel, size=label_size)

    if legend_title:
        ax.legend(title=legend_title, fontsize=12, title_fontsize=14)

    return xticks
