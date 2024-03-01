import attrs
import numpy as np
import pandas as pd

from .basis import MolecularOrbitalBasis

__all__ = [
    'LabelByPartition',
]


@attrs.define(repr=True, eq=True, frozen=True)
class LabelByPartition:
    by: list[str]
    n_top: int
    save: bool = False
    return_result: bool = True

    @property
    def n_by(self) -> int:
        return len(self.by)

    def __call__(self, mo: MolecularOrbitalBasis) -> pd.DataFrame | None:
        df = mo.get_partitioning(by=self.by)
        fragments, partorb = df.iloc[:, :self.n_by], df.iloc[:, self.n_by:]
        fragments['label'] = fragments.apply(lambda r: '_'.join(r), axis=1)

        partorb_abs = partorb.apply(np.abs)
        indicies = np.argpartition(partorb_abs, -self.n_top, axis=0)
        labels = fragments['label'].values[indicies[-self.n_top:, :]][::-1, :]
        coeffs = np.take_along_axis(partorb.to_numpy(), indicies[-self.n_top:, :], 0)[::-1, :]

        df_labels = pd.DataFrame(labels.T, columns=[f'label_top{i}' for i in range(1, self.n_top + 1)])
        df_coeffs = pd.DataFrame(coeffs.T, columns=[f'coeff_top{i}' for i in range(1, self.n_top + 1)])
        df = pd.concat(
            [pd.concat([df_labels.iloc[:, i], df_coeffs.iloc[:, i]], axis=1) for i in range(self.n_top)], axis=1)

        if self.save:
            mo.df = pd.concat([mo.df, df], axis=1)

        if self.return_result:
            return df
