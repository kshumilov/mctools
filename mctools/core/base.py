from __future__ import annotations

import abc

from typing import NoReturn, Callable, Sequence

import pandas as pd
import numpy as np
import numpy.typing as npt

__all__ = [
    'Consolidator',
    'Selector',
]


Selector = Callable[[pd.DataFrame], bool]


class Consolidator(abc.ABC):
    """Defines the Interface for a class that consolidates a group of entries
    and their respective properties into a table.

    The table is stored in the Pandas DataFrame format. The columns of the
    dataframe are divided into two kinds:
        - IDX_COLS: Columns that are used to uniquely identify entries in the
            table. Usually includes the SOURCE_COL — the name of the file
            from which the data was obtained, and some additional columns.
        - Property columns: All other columns that are present in the dataframe,
            but are not used for indexing, and a result of additional
            calculations or data manipulations.
        - DEFAULT_COLS: Property columns that must be present in the dataframe,
            for it to be valid.

    Class Attributes:
        IDX_NAME: name of the index in the _df.
        SOURCE_COL: name of the colum defining the source on an entry.
        IDX_COLS: See above.
        DEFAULT_COLS: See above.

    Attributes:
        _df: The table is stored in the Pandas DataFrame format.
    """
    IDX_NAME = 'idx'
    SOURCE_COL = 'source'

    IDX_COLS: list[str] = []
    DEFAULT_COLS: list[str] = []

    __slots__ = [
        '_df',
    ]

    _df: pd.DataFrame

    def __init__(self: 'Consolidator', df: pd.DataFrame, *args, source: str = '', sort: bool = False, **kwargs) -> None:
        if self.SOURCE_COL not in df:
            if source:
                df[self.SOURCE_COL] = source
            else:
                raise ValueError(f"either 'df' must have {self.SOURCE_COL} or "
                                 f"source argument must be passed to {self.__class__.__name__}")

        self.df = df
        self.reset_index()

        if sort:
            self.sort()

    @abc.abstractmethod
    def analyze(self: 'Consolidator', idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False) -> pd.DataFrame | None:
        pass

    def sort(self: 'Consolidator', col: str = '') -> NoReturn:
        if col in self._df:
            self._df.sort_values(col, ignore_index=True, inplace=True)

        self.reset_index()

    def filter(self: 'Consolidator', idx: npt.ArrayLike | None = None, condition: Selector | None = None,
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
    def df(self: 'Consolidator') -> pd.DataFrame:
        return self._df

    @df.setter
    def df(self, new_df: pd.DataFrame) -> NoReturn:
        self._df = self.validate_df(new_df)

    def validate_df(self: 'Consolidator', new_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.DEFAULT_COLS:
            if col not in new_df:
                raise ValueError(f"'df' must have {col}")
        return new_df

    def add_properties(self: 'Consolidator', data: npt.ArrayLike, names: str | Sequence[str], replace=True) -> pd.DataFrame:
        data = np.asarray(data)

        if data.ndim == 0:
            data = np.full(len(self), data)
            names = [names]

        if data.ndim > 2 and len(data) != len(self) and data.shape[-1] != len(names):
            raise ValueError(f"'data' must of the same length as {self!r}: {len(self)}")

        data = data.reshape(len(self), -1)
        df = pd.DataFrame(data, columns=names)
        self.update_properties(df, replace=replace)
        return df

    def update_properties(self: 'Consolidator', new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)  # Columns on new_df that can be added
        duplicate_cols = cols & self.property_columns  # Columns that exist in both new_df and self.df

        if replace:
            self._df.drop(columns=list(duplicate_cols), inplace=True)
            duplicate_cols.clear()

        new_cols = cols - duplicate_cols
        self._df = pd.concat([self._df, new_df[list(new_cols)]], axis=1, copy=False)
        self._df.update(new_df[list(duplicate_cols)])

    def clear_properties(self: 'Consolidator') -> NoReturn:
        self._df.drop(columns=list(self.property_columns), inplace=True)

    @property
    def property_columns(self: 'Consolidator') -> set[str]:
        """Property columns that exist on self.df"""
        return set(self._df.columns) - set(self.DEFAULT_COLS)

    def reset_index(self: 'Consolidator') -> NoReturn:
        self._df.reset_index(drop=True, inplace=True)
        self._df.index.name = self.IDX_NAME

    def __len__(self: 'Consolidator') -> int:
        return len(self._df)
