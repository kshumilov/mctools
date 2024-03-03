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

    def analyze(self, mo: MolecularOrbitalBasis) -> pd.DataFrame:
        df = mo.get_partitioning(by=self.by)
        fragments, partorb = df.iloc[:, :self.n_by], df.iloc[:, self.n_by:]
        fragments['label'] = fragments.apply(lambda r: '_'.join(r), axis=1)

        top_n = np.s_[-self.top:, :]
        inverted = np.s_[::-1, :]
        indicies = np.argpartition(partorb.transform(np.abs), -self.top, axis=0)
        labels = fragments['label'].values[indicies[top_n]][inverted]
        coeffs = np.take_along_axis(partorb.to_numpy(), indicies[top_n], 0)[inverted]
        rest = np.take_along_axis(partorb.to_numpy(), indicies[:-self.top, :], 0).sum(axis=0)

        ordering = coeffs.argsort(axis=0)[::-1, :]
        labels = np.take_along_axis(labels, ordering, axis=0)
        coeffs = np.take_along_axis(coeffs, ordering, axis=0)

        df_labels = pd.DataFrame(labels.T, columns=[f'label_top{i}' for i in range(1, self.top + 1)])
        df_coeffs = pd.DataFrame(coeffs.T, columns=[f'coeff_top{i}' for i in range(1, self.top + 1)])

        df = pd.concat(
            [
                pd.concat([df_labels.iloc[:, i], df_coeffs.iloc[:, i]], axis=1)
                for i in range(self.top)
            ],
            axis=1
        )

        df['rest_coeff'] = rest

        return df


