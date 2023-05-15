from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .molecule import Molecule
    from ..parser.lib import ParsingResult


__all__ = [
    'Basis',
]


class Basis:
    __slots__ = [
        'molecule'
    ]

    molecule: Molecule

    @classmethod
    def from_data(cls, data: ParsingResult):
        pass
