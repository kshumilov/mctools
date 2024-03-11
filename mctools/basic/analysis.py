import attrs
import numpy as np
import pandas as pd

from mctools.newcore.analyzer import Analyzer
from .basis import MolecularOrbitalBasis

__all__ = [
    'LabelByPartition',
]


@attrs.define(repr=True, eq=True, frozen=True)
class LabelByQuery(Analyzer):
    def analyze(self, mo: MolecularOrbitalBasis) -> pd.DataFrame:
        raise NotImplementedError()


@attrs.define(repr=True, eq=True, frozen=True)
class LabelByPartition(Analyzer):
    by: list[str] = attrs.field(factory=lambda: ['atom'])
    top: int = 1

    @property
    def n_by(self) -> int:
        return len(self.by)

    def analyze(self, c: MolecularOrbitalBasis) -> pd.DataFrame:
        partorb = c.get_partition(self.by)
        labels = partorb.index.map('-'.join).to_numpy(dtype='U')

        indices = np.argpartition(partorb.transform(np.abs), -self.top, axis=0)

        mo_labels = labels[indices[-self.top:, :]][::-1, :]
        mo_coeffs = np.take_along_axis(partorb.to_numpy(), indices[-self.top:, :], 0)[::-1, :]
        rest = np.take_along_axis(partorb.to_numpy(), indices[:-self.top, :], 0).sum(axis=0)

        ordering = mo_coeffs.argsort(axis=0)[::-1, :]
        mo_labels = np.take_along_axis(mo_labels, ordering, axis=0)
        mo_coeffs = np.take_along_axis(mo_coeffs, ordering, axis=0)

        df_labels = pd.DataFrame(mo_labels.T, columns=[f'label_top{i}' for i in range(1, self.top + 1)])
        df_coeffs = pd.DataFrame(mo_coeffs.T, columns=[f'coeff_top{i}' for i in range(1, self.top + 1)])

        df = pd.concat(
            [
                pd.concat([df_labels.iloc[:, i], df_coeffs.iloc[:, i]], axis=1)
                for i in range(self.top)
            ],
            axis=1
        )

        df['coeff_rest'] = rest

        return df


