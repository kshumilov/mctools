import pandas as pd

from .molecule import Molecule


__all__ = [
    'Basis',
]


class Basis:
    __slots__ = [
        'shells',
        'primitives',

        'molecule'
    ]

    shells: pd.DataFrame
    primitives: pd.DataFrame

    molecule: Molecule
