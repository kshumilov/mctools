from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt

from ..core import MCStates

__all__ = [
    'prepare_state_centers',
    'plot_state_levels',
]


def prepare_state_centers(states: MCStates, /,
                          center: float = 0., *,
                          max_col_width: float = 1.,
                          degeneracy_col: str = MCStates.DEGENERACY_COL) -> np.ndarray:
    E = states.E
    g = states.df[degeneracy_col]

    x_c = np.zeros_like(E) + (center - max_col_width / 2)
    for d, n in zip(*np.unique(g, return_counts=True)):
        l = max_col_width / n
        j = 2 * np.arange(n) + 1
        x_c[(g == d)] += j * l / 2
    return x_c


def plot_state_levels(
        ax: plt.Axes,

        data: MCStates | Sequence[MCStates],
        energy_col: str = MCStates.RELATIVE_ENERGY_COL,
        degeneracy_col: str = MCStates.DEGENERACY_COL,

        title: str | None = None,
        xlabel: str = 'Structure',
        ylabel: str = r'$E-E_0$',
        legend_title: str = 'States',

        state_labels: dict[str, str] | None = None,
        xlabels: Sequence[str] | None = None,
        state_label_col: str = MCStates.DOMINANT_CONFIG_COL,

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
    for center, states in enumerate(data):
        center *= center_scale
        x_c = prepare_state_centers(states, center=center,
                                    max_col_width=column_width,
                                    degeneracy_col=degeneracy_col)
        xticks.append(center)

        if MCStates.DOMINANT_CONFIG_COL in states.property_columns:
            iterator = states.df.groupby(state_label_col)
        else:
            iterator = zip([None], [states.df])

        for i, (label, df) in enumerate(iterator):
            E = df[energy_col]
            level_end = x_c[df.index.values] + (level_width / 2)
            level_start = x_c[df.index.values] - (level_width / 2)

            label = state_labels.get(label, label) if center == 0 else None
            ax.hlines(E, level_start, level_end, label=label,
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
