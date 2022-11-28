import numpy as np

from ..core import MCStates


def prepare_state_centers(states: MCStates, center: float = 0., column_width: float = 1.) -> np.ndarray:
    E = states.df[states.E_COL]
    g = states.df['g']

    x_c = np.zeros_like(E) + (center - column_width / 2)
    for d, n in zip(*np.unique(g, return_counts=True)):
        idx = (g == d)
        l = column_width / n
        j = 2 * np.arange(n) + 1
        x_c[idx] += j * l / 2
    return x_c
