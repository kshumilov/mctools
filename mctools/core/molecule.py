from __future__ import annotations

from typing import Any, NoReturn, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from .base import Consolidator, Selector


if TYPE_CHECKING:
    from ..parser.lib import ParsingResult


__all__ = [
    "Molecule",
]


class Molecule(Consolidator):
    """Class to hold basic information about the molecule

    Attributes:
        df: pd.DataFrame --- stores definitions about atoms;
        charge: int --- charge on the molecules
        multiplicity: int --- M_s = 2 * S + 1, where S is sum of all spins of the electrons

        name: str --- name of the molecule, optional
        source: str --- filename of the origin
    """
    __slots__ = [
        "charge", "multiplicity",
    ]

    # Cartesian Coordinates
    X_COL = 'x'
    Y_COL = 'y'
    Z_COL = 'z'
    COORDS_COLS = [X_COL, Y_COL, Z_COL]

    # Atom properties
    ATOM_COL = 'atom'
    ATOMIC_NUMBER_COL = 'Z'

    SOURCE_COL = 'atom_source'
    IDX_COLS = [ATOM_COL, SOURCE_COL]
    DEFAULT_COLS = [*IDX_COLS, *COORDS_COLS, ATOMIC_NUMBER_COL]

    _df: pd.DataFrame  # Table that hold properties of atoms in the molecule
    name: str | None

    def __init__(self, df: pd.DataFrame, /,
                 source: str | None = None,
                 charge: int = 0,
                 multiplicity: int = 1):
        super(Molecule, self).__init__(df, source=source, sort=False)
        self.charge = charge
        if charge > self.Z.sum():
            raise ValueError('Cannot ionize more electrons then given in atoms')

        self.multiplicity = multiplicity

    def analyze(self: 'Consolidator', idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False) -> pd.DataFrame | None:
        pass

    @property
    def coords(self: 'Molecule') -> npt.NDArray[np.float_]:
        return self._df[self.COORDS_COLS].values
    
    @property
    def Z(self: 'Molecule') -> npt.NDArray[np.int_]:
        return self._df[self.ATOMIC_NUMBER_COL].values
    
    @property
    def n_elec(self: 'Molecule') -> np.int_:
        return self.Z.sum() - self.charge
    
    @classmethod
    def from_dict(cls, data: ParsingResult, /,
                  df_key: str = 'df_molecule',
                  charge_key: str = 'charge',
                  multiplicity_key: str = 'multiplicity',
                  source_key: str = 'source',
                  instance_key: str = 'molecule',
                  **kwargs) -> 'Molecule':
        if isinstance(instance := data.get(instance_key, None), cls):
            return instance
        elif isinstance(instance, dict):
            instance = cls.from_dict(instance, **kwargs)
        elif instance is None:
            data.update(kwargs)

            df = data.pop(df_key)
            source = data.get(source_key, '')

            args: list[Any] = []
            if charge := data.pop(charge_key, None):
                args.append(charge)

            if multiplicity_key := data.pop(multiplicity_key, None):
                args.append(multiplicity_key)

            instance = data.setdefault('molecule',
                                       cls(df, source, *args))
        else:
            raise ValueError(f"{cls.__name__} did not recognized '{instance_key}' "
                             f"item in data: {instance}")
        return instance

    def validate_df(self: 'Molecule', new_df: pd.DataFrame) -> pd.DataFrame:
        if self.ATOM_COL not in new_df:
            new_df[self.ATOM_COL] = np.arange(len(new_df)) + 1

        return super(Molecule, self).validate_df(new_df)

    def to_xyz(self: 'Molecule', filename: str) -> NoReturn:
        lines = ['%4.d' % len(self), '']
        for idx, atom in self.df.iterrows():
            line = ['%4.d' % atom.Z]
            line.extend(['%16.8E' % v for v in atom[self.COORDS_COLS]])
            lines.append((' ' * 4).join(line))

        with open(filename, 'w') as file:
            file.write('\n'.join(lines))

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        name = ''
        return f"{self.__class__.__name__}" \
               f"({name}, #atoms={len(self)})"
    