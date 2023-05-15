import numpy as np

from ..core import MCStates


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
