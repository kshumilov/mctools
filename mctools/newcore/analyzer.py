from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TypeVar, Generic

import attrs
import pandas as pd

from .consolidator import Consolidator

__all__ = [
    'Analyzer',
]


C = TypeVar('C', bound=Consolidator, contravariant=True)


@attrs.define(eq=True, repr=True, frozen=True)
class Analyzer(Generic[C], metaclass=ABCMeta):
    def __call__(self, c: C, /, save: bool = False) -> pd.DataFrame | None:
        df = self.analyze(c)

        if save:
            c.df = pd.concat([c.df, df], axis=1)

        return df

    @abstractmethod
    def analyze(self, consolidator: C) -> pd.DataFrame:
        raise NotImplementedError()
