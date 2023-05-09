from __future__ import annotations

import abc

from typing import NoReturn, Callable

import pandas as pd
import numpy as np
import numpy.typing as npt

__all__ = [
    'MCBase',
    'Selector'
]


Selector = Callable[[pd.DataFrame], bool]


class MCBase(abc.ABC):
    IDX_NAME = 'idx'
    SOURCE_COL = 'source'

    IDX_COLS: list[str] = []
    DEFAULT_COLS: list[str] = []

    __slots__ = [
        '_df',
    ]

    _df: pd.DataFrame

    @abc.abstractmethod
    def analyze(self: 'MCBase', idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False) -> pd.DataFrame | None:
        pass

    def sort(self: 'MCBase', col: str = '') -> NoReturn:
        if col in self._df:
            self._df.sort_values(col, ignore_index=True, inplace=True)

        self.reset_index()

    def filter(self: 'MCBase', idx: npt.ArrayLike | None = None, condition: Selector | None = None,
               label_index: bool = False) -> np.ndarray | pd.Index:
        """Filter entries on positional integer index and some condition.

        Notes:
            Entries are selected based on provided index first and then based on condition. If no `idx` or
            `condition` is provided, the function returns array of indices to all states.

        Keyword Args:
            idx: Sequence of states indices to which cond() will be applied.
                If idx is None, cond() is applied to all states.
            cond: Predicate to filter states by, if not provided, no filtering is done. Cond() must take Dataframe as
                its argument and return Sequence of booleans equal in length to the length to the provided Dataframe.
            label_index: flag indicating whether DataFrame's index is returned or np.ndarray is returned

        Returns:
            Numpy 1D array with indices of selected states or there associated labels in DataFram

        Future Developments:
            TODO: return MCBase object
        """
        idx = np.asarray(idx).reshape(-1) if idx is not None else np.arange(len(self))
        selected = condition(self._df.iloc[idx]) if condition else np.s_[...]

        if label_index:
            return self._df.index[selected]

        return idx[selected]

    @property
    def df(self: 'MCBase') -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new_df: pd.DataFrame) -> NoReturn:
        self._df = self.validate_df(new_df)

    def validate_df(self: 'MCBase', new_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.DEFAULT_COLS:
            if col not in new_df:
                raise ValueError(f"'df' must have {col}")

        return new_df

    def update_properties(self: 'MCBase', new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)  # Columns on new_df that can be added
        duplicate_cols = cols & self.property_columns  # Columns that exist in both new_df and self.df

        if replace:
            self._df.drop(columns=list(duplicate_cols), inplace=True)
            duplicate_cols.clear()

        new_cols = cols - duplicate_cols
        self._df = pd.concat([self._df, new_df[list(new_cols)]], axis=1, copy=False)
        self._df.update(new_df[list(duplicate_cols)])

    def clear_properties(self: 'MCBase') -> NoReturn:
        self._df.drop(columns=list(self.property_columns), inplace=True)

    @property
    def property_columns(self: 'MCBase') -> set[str]:
        """Property columns that exist on self.df"""
        return set(self._df.columns) - set(self.DEFAULT_COLS)

    def reset_index(self: 'MCBase') -> NoReturn:
        self._df.reset_index(drop=True, inplace=True)
        self._df.index.name = self.IDX_NAME

    def __len__(self: 'MCBase') -> int:
        return len(self._df)
