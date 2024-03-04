from __future__ import annotations

import attrs
import numpy as np
import pandas as pd

from .states import States


@attrs.define(repr=True, eq=True, frozen=True)
class PartitionRDMDiagonals:
    label_col: str

    save: bool = False
    return_result: bool = True

    def __call__(self, states: States) -> pd.DataFrame:
        rdm_diags = states.rdms.diagonal(axis1=1, axis2=2)

        mo_labels = states.mobasis.df[states.active_idx, self.label_col]
        uniques = np.unique(mo_labels)
        df = pd.DataFrame(rdm_diags, columns=mo_labels.columns)
        df.groupby(self.label_col)[uniques].sum()

        if self.save:
            states.df = pd.concat([states.df, df], axis=1)

        if self.return_result:
            return df
