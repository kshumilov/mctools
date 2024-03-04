from __future__ import annotations

import attrs
import numpy as np
import pandas as pd

from .states import States
from ..newcore.analyzer import Analyzer

__all__ = [
    'PartitionRDMDiagonals',
]


@attrs.define(repr=True, eq=True, frozen=True)
class PartitionRDMDiagonals(Analyzer[States]):
    mo_label_col: str = 'active_space'

    def analyze(self, consolidator: States) -> pd.DataFrame:
        col_idx = consolidator.mobasis.df.columns.get_loc(self.mo_label_col)
        mo_labels = consolidator.mobasis.df.iloc[consolidator.active_idx, col_idx]

        rdm_diags = consolidator.rdms.diagonal(axis1=1, axis2=2)
        dfr = pd.DataFrame(rdm_diags, columns=mo_labels)
        return dfr.T.groupby(self.mo_label_col).sum().T
