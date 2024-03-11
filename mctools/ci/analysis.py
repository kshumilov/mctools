from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
import pandas as pd

from ..newcore.analyzer import Analyzer

if TYPE_CHECKING:
    from .states import States
    from .transitions import Transitions

__all__ = [
    'PartitionRDMDiagonals',
    'CalculateTransitionEnergy',
]


@attrs.define(repr=True, eq=True, frozen=True)
class PartitionRDMDiagonals(Analyzer[States]):
    mo_label_col: str = 'active_space'

    def analyze(self, c: States) -> pd.DataFrame:
        col_idx = c.mobasis.df.columns.get_loc(self.mo_label_col)
        mo_labels = c.mobasis.df.iloc[c.active_idx, col_idx]

        dfr = pd.DataFrame(
            c.rdms.diagonal(axis1=1, axis2=2),
            columns=mo_labels,
            index=c.df.index,
        )
        return dfr.T.groupby(self.mo_label_col).sum().T


@attrs.define(repr=True, eq=True, frozen=True)
class CalculateTransitionEnergy(Analyzer[Transitions]):
    energy_col: str = 'E'

    def analyze(self, c: Transitions) -> pd.DataFrame:
        df_E = c.get_state_properties([self.energy_col])
        return pd.DataFrame({'dE': df_E['E_fdx'] - df_E['E_idx']})


@attrs.define(repr=True, eq=True, frozen=True)
class CalculateRDMDiagonalDifference(Analyzer[Transitions]):
    mo_label_col: str = 'active_space'

    def analyze(self, c: Transitions) -> pd.DataFrame:
        states = c.states

        col_idx = states.mobasis.df.columns.get_loc(self.mo_label_col)
        mo_labels = np.unique(states.mobasis.df.iloc[states.active_idx, col_idx])
        dfr = c.get_state_properties(mo_labels)

        df_dict = {col: dfr[f'{col}_fdx'] + dfr[f'{col}_idx'] for col in mo_labels}
        return pd.DataFrame.from_dict(df_dict)
