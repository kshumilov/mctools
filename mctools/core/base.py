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

    df_: pd.DataFrame

    @abc.abstractmethod
    def analyze(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False) -> pd.DataFrame | None:
        pass

    def sort(self, col: str = '') -> NoReturn:
        if col in self.df_:
            self.df_.sort_values(col, ignore_index=True, inplace=True)

        self.reset_index()

    def filter(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
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
        selected = condition(self.df_.iloc[idx]) if condition else np.s_[...]

        if label_index:
            return self.df_.index[selected]

        return idx[selected]

    def update_properties(self, new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)  # Columns on new_df that can be added
        duplicate_cols = cols & self.property_columns  # Columns that exist in both new_df and self.df

        if replace:
            self.df_.drop(columns=duplicate_cols, inplace=True)
            duplicate_cols.clear()

        new_cols = cols - duplicate_cols
        self.df_ = pd.concat([self.df_, new_df[list(new_cols)]], axis=1, copy=False)
        self.df_.update(new_df[list(duplicate_cols)])

    def clear_properties(self) -> NoReturn:
        self.df_.drop(columns=self.property_columns, inplace=True)

    @property
    def property_columns(self) -> set[str]:
        """Property columns that exist on self.df"""
        return set(self.df_.columns) - set(self.DEFAULT_COLS)

    def reset_index(self) -> NoReturn:
        self.df_.reset_index(drop=True, inplace=True)
        self.df_.index.name = self.IDX_NAME

    def __len__(self) -> int:
        return len(self.df_)
