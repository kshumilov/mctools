from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = [
    "Molecule",
]


class Molecule:
    """Class to hold basic information about the molecule

    Attributes:
        df: pd.DataFrame --- stores definitions about atoms;
        charge: int --- charge on the molecules
        multiplicity: int --- M_s = 2 * S + 1, where S is sum of all spins of the electrons

        name: str --- name of the molecule, optional
        source: str --- filename of the origin
    """
    __slots__ = [
        "df",
        "charge", "multiplicity",
        "name", "source",
    ]

    ## Columns definitions
    SOURCE_COL = 'source'

    # Cartesian Coordinates
    X_COL = 'x'
    Y_COL = 'y'
    Z_COL = 'z'
    COORDS_COLS = [X_COL, Y_COL, Z_COL]

    # Atom properties
    ATOMIC_NUMBER_COL = 'Z'

    DEFAULT_COLS = [*COORDS_COLS, ATOMIC_NUMBER_COL]

    _df: pd.DataFrame  # Table that hold properties of atoms in the molecule
    name: str | None

    def __init__(self, molecule: pd.DataFrame, *, 
                 charge: int = 0, multiplicity: int = 1, /,
                 name: str | None = None, source: str | None = None):
        for col in self.DEFAULT_COLS:
            if col not in molecule:
                raise ValueError(f"'molecule' DataFrame must contain {col} column.")

        self._df = molecule.copy(deep=True)
        self.charge = charge
        self.multiplicity = multiplicity
        
        self.name = name
        self.source = source

    @property
    def df(self: 'Molecule') -> pd.DataFrame:
        return self.df.copy(deep=True)

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
    def from_dict(cls, data: dict[str, Any], /,
                  coords_key: str = 'coords',
                  atomic_number_key: str = 'Z',
                  charge_key: str = 'charge',
                  multiplicity_key: str = 'multiplicity',
                  source_key: str = 'source',
                  name_key: str = 'title', **kwargs) -> 'Molecule':
        data.update(kwargs)

        args: list[Any] = []

        df = pd.DataFrame()
        df[cls.COORDS_COLS] = data.pop(coords_key)
        df[cls.ATOMIC_NUMBER_COL] = data.pop(atomic_number_key)
        args.append(df)

        if charge_key in data:
            args.append(data.pop(charge_key))

        if multiplicity_key in data:
            args.append(data.pop(multiplicity_key))

        return cls(*args, name=data.pop(name_key), source=data.pop(source_key))

    def __len__(self) -> int:
        return len(self._df)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}" \
               f"({self.name}, #atoms={len(self)}, source='{self.source}')"
    