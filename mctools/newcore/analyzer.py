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
    save: bool = False
    return_result: bool = True

    def __call__(self, consolidator: C) -> pd.DataFrame | None:
        df = self.analyze(consolidator)

        if self.save:
            consolidator.df = pd.concat([consolidator.df, df], axis=1)
        if self.return_result:
            return df

        return None

    @abstractmethod
    def analyze(self, consolidator: Consolidator) -> pd.DataFrame:
        raise NotImplementedError()
