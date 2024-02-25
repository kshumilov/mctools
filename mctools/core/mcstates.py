from __future__ import annotations

import warnings

from typing import NoReturn, TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy import sparse

from .base import Consolidator
from .utils.constants import Eh2eV
from .mcspace import MCSpace

if TYPE_CHECKING:
    from .mctransitions import MCTransitions
    from .base import Selector
    from .mcspace import ConfigTransform
    from parsing.core.pattern import ParsingResultType

__all__ = [
    'MCStates',
]


class MCStates(Consolidator):
    """Holds information about multiconfigurational states.

    Attributes:
        space: CAS/RAS definition, and related utilities.
        df: Energy (in Eh) of states, and any other information that can be associated
            with a particular state. States are indexed internally by df.index starting from 0, with unique index
            defined by pair of (STATE_COL, SOURCE_COL). df.shape = (#States, #Properties)
        ci_vecs: CI vectors as in Sparse Compressed Row format. ci_vec.shape = (#States, #Determinants)
        rdm_diags: Diagonals of 1RDM matrices for every state. rdm_diag.shape = (#States, #Active MOs)

    Possible columns on df:
        STATE_COL: index of the state as defined in the `source`;
        ENERGY_COL: Energy of the state in Hartree;

    Future Development:
        TODO: add name attribute
        TODO: Explore LIL sparse structure for manipulation of ci_vec
        TODO: pd.MultiIndex for permanent cols, MO blocks, config classes, and other computed properties
        TODO: Adapt to using dense representation of CI vectors
        TODO: try using STATE_COL and SOURCE_COL as index in df
        TODO: include duplicate and similar
        TODO: Move _state_map to RESOURCE_COL
    """
    ENERGY_COL = 'E'
    RELATIVE_ENERGY_COL = 'E0'
    DEGENERACY_COL = 'g'
    STATE_COL = 'state'
    CI_NORM_COL = 'norm'
    DOMINANT_CONFIG_COL = 'config_class'
    SOURCE_COL = f'{STATE_COL}_source'
    RESOURCE_COL = 'resource_idx'  # index of related resource in ci_vecs and rdm_diag, see _state_map

    IDX_COLS = [STATE_COL, SOURCE_COL]  # Columns used to identify the states uniquely
    DEFAULT_COLS = [*IDX_COLS, ENERGY_COL]  # Permanent property columns for the df

    __slots__ = (
        'ci_vecs',
        'rdm_diags',

        'mcspace',
        'transitions',
        '_state_map'
    )

    ci_vecs: sparse.csr_array | sparse.lil_array  # Sparse array of CI vectors
    rdm_diags: np.ndarray

    # mcspace: MCSpace | None
    # peaks: MCPeaks | None

    # Used for implicit sorting of rdm_diag and ci_vec
    # Move to df
    _state_map: np.ndarray

    def __init__(self, df: pd.DataFrame,
                 ci_vecs: sparse.csr_array | sparse.lil_array | sparse.coo_array,
                 rdm_diags: np.ndarray, /,
                 source: str = '',
                 mcspace: MCSpace | None = None, *,
                 sort: bool = False) -> None:
        self.ci_vecs = ci_vecs.tocsr()
        self.rdm_diags = rdm_diags

        super(MCStates, self).__init__(df, source=source, sort=False)

        if mcspace is not None:
            if mcspace.n_mo_act != rdm_diags.shape[1]:
                raise ValueError("Number of active MOs in MCSpace must be equal to number of MOs in 'rdm_diags' array")

            if mcspace.n_configs != ci_vecs.shape[1]:
                raise ValueError("Number of configurations in MCSpace must be equal to number of determinants in "
                                 "'ci_vecs' array")

        self.mcspace = mcspace
        self._state_map = np.arange(len(self))
        self.transitions: MCTransitions | None = None
        # self.df['resource_idx'] = self._state_map

        if sort:
            self.sort(self.ENERGY_COL)

    def sort(self, col: str = ENERGY_COL) -> NoReturn:
        idx = np.argsort(self._df[col].values)
        self._state_map = self._state_map[idx]
        self._df = self._df.iloc[idx]

        self.reset_index()
        if self.transitions is not None:
            if self.transitions.INITIAL_COL in self.transitions.df or self.transitions.FINAL_COL in self.transitions.df:
                self.transitions.calculate_state_idx(save=True, replace=True)

    def analyze(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False, **kwargs) -> pd.DataFrame | None:
        """Performs analysis of states, using the following functions:
            - estimate_state_degeneracy
            - calculate_ci_vec_norm
            - calculate_relative_energy
            - partition_rdm_diag
            - partition_ci_vec
            - estimate_dominant_config_class

        Notes:
            Before performing the analysis all previously calculated properties are cleared.

        Args:
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            condition:  Predicate to select state that are analyzed. See .filter() method.
            save: Whether to save result locally (save=True) onto self.df using self.update_properties(),
                or return calculated result directly without changing local df (False).
            replace: If save=True, choose to either append new properties as new columns (replace=False) or replace old
                columns with new result if duplicate columns are found (replace=False).
            **kwargs:

        Keyword Args:
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            cond: Predicate to select state that are analyzed. See .filter() method.
            save: If True, saves result
            replace: if results are saved, whether to replace old results with new, or to store
            **kwargs:

        Returns:
            If result is not saved on the Dataframe, return the Dataframe with the results
        """
        self.clear_properties()

        dfs = [
            self.estimate_state_degeneracy(**kwargs, idx=idx, condition=condition, save=save, replace=replace),
            self.calculate_relative_energy(**kwargs, idx=idx, condition=condition, save=save, replace=replace),
            self.calculate_ci_vec_norm(**kwargs, idx=idx, condition=condition, save=save, replace=replace),
        ]

        try:
            dfs.append(self.partition_rdm_diag(idx=idx, condition=condition, save=save, replace=replace))
            dfs.append(self.partition_ci_vec(idx=idx, condition=condition, save=save, replace=replace))
            dfs.append(self.estimate_dominant_config_class(idx=idx, condition=condition, save=save, replace=replace))
        except ValueError as err:
            warnings.warn(err.args[0])

        if not save:
            return pd.concat([self._df[self.ENERGY_COL], *dfs], axis=1)

        if self.transitions is not None:
            self.transitions.analyze(save=True)

    def partition_rdm_diag(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None, *,
                           save: bool = True, replace: bool = False, **kwargs) -> pd.DataFrame | None:
        """Partitions RDM Diagonal for selected states based on MO blocks, defined in MCSpace.

        Notes:
            Filtering of states based on index and cond() can be performed initially.

        Args:
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            condition: Predicate to select state that are analyzed. See .filter() method.
            save: Whether to save result locally (save=True) onto self.df using self.update_properties(),
                or return calculated result directly without changing local df (False).
            replace: If save=True, choose to either append new properties as new columns (replace=False) or replace old
                columns with new result if duplicate columns are found (replace=False).
            **kwargs: N/A

        Returns:
            if save=True:
                pd.DataFrame with calculated partitioned RDM diagonal for selected states
            else:
                None
        """
        if not self.is_space_set:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)

        df = self.mcspace.partition_rdm_diag(self.rdm_diags[self._state_map[idx]])
        df.set_index(self._df.index[idx], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def partition_ci_vec(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None, *,
                         save: bool = True, replace: bool = False, **kwargs) -> pd.DataFrame | None:
        """Partitions CI vectors for selected states based on Configuration classes, defined in MCSpace.

        Notes:
            Filtering of states based on index and cond() can be performed initially.

        Args:
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            condition: Predicate to select state that are analyzed. See .filter() method.
            save: Whether to save result locally (save=True) onto self.df using self.update_properties(),
                or return calculated result directly without changing local df (False).
            replace: If save=True, choose to either append new properties as new columns (replace=False) or replace old
                columns with new result if duplicate columns are found (replace=False).
            **kwargs:  N/A

        Returns:
            if save=True:
                pd.DataFrame with calculated partitioned CI vectors for selected states.
            else:
                None
        """
        if not self.is_space_set:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)

        df = self.mcspace.partition_ci_vec(self.ci_vecs[self._state_map[idx]])
        df.set_index(self._df.index[idx], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_ci_vec_norm(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                              save: bool = True, replace: bool = False, col_name: str = CI_NORM_COL, **kwargs):
        """Calculates norm of CI vector for selected states.

        norm = sqrt(Sum_i[C_i^h * C_i * <Det_i|Det_i>])

        Notes:
            Filtering of states based on index and cond() can be performed initially.

        Args:
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            condition: Predicate to select state that are analyzed. See .filter() method.
            save: Whether to save result locally (save=True) onto self.df using self.update_properties(),
                or return calculated result directly without changing local df (False).
            replace: If save=True, choose to either append new properties as new columns (replace=False) or replace old
                columns with new result if duplicate columns are found (replace=False).
            col_name: name of the column to store the result in.
            **kwargs: N/A

        Returns:
            if save=True:
                pd.DataFrame with calculated partitioned CI vector norms for selected states.
            else:
                None
        """
        if not self.is_space_set:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)
        norm = sparse.linalg.norm(self.ci_vecs[self._state_map[idx]], axis=1)

        df = pd.DataFrame({col_name: norm}, index=self._df.index[idx])

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def estimate_state_degeneracy(self, tol: float = 1e-3,
                                  idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                                  save: bool = True, replace: bool = True,
                                  col_name: str = DEGENERACY_COL, **kwargs) -> pd.DataFrame | None:
        """Estimates state degeneracy based on state energy and assign degeneracy class, g. States with the same
         degeneracy class are considered the degenerate.

        Args:
            tol: Energy tolerance, in Hartree, above which states are considered different.
            idx: Sequence of states indices that are passed to cond() before analysis, see .filter() method.
            condition: Predicate to select state that are analyzed. See .filter() method.
            save: Whether to save result locally (save=True) onto self.df using self.update_properties(),
                or return calculated result directly without changing local df (False).
            replace: If save=True, choose to either append new properties as new columns (replace=False) or replace old
                columns with new result if duplicate columns are found (replace=False).
            col_name:

        Returns:
            if save=True:
                pd.DataFrame with calculated degeneracies for selected states.
            else:
                None
        """
        idx = self.filter(idx=idx, condition=condition)

        dE = self._df.loc[self._df.index[idx], self.ENERGY_COL].diff()
        dE.fillna(0, inplace=True)
        g = (dE > tol).cumsum()

        df = pd.DataFrame({col_name: g})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def estimate_dominant_config_class(self: 'MCStates',
                                       idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                                       save: bool = True, replace: bool = False,
                                       col_name: str = DOMINANT_CONFIG_COL, **kwargs) -> pd.DataFrame | None:
        if not (self.is_space_set and self.mcspace.are_config_classes_set):
            raise ValueError('Set configuration classes on MCSpace first.')

        idx = self.filter(idx=idx, condition=condition)

        config_columns = list(self.property_columns & self.mcspace.config_classes.keys())
        config_class = self._df.loc[self._df.index[idx], config_columns].idxmax(axis=1).values

        df = pd.DataFrame({col_name: config_class})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_relative_energy(self: 'MCStates', scale: float = Eh2eV, E_min: float | None = None,
                                  idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                                  save: bool = True, replace: bool = True,
                                  col_name: str = RELATIVE_ENERGY_COL, **kwargs) -> pd.DataFrame | None:
        idx = self.filter(idx=idx, condition=condition)

        E0 = self._df.loc[self._df.index[idx], self.ENERGY_COL]
        E_min = E_min if E_min is not None else E0.min()
        E0 -= E_min
        E0 *= scale

        df = pd.DataFrame({col_name: E0})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def find_similar(self, other: 'MCStates', /, ignore_space: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not ignore_space and not (self.is_space_set and other.is_space_set):
            warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')

        if not ignore_space and self.mcspace != other.mcspace:
            raise ValueError('spaces of states are different, overlap is poorly defined')

        self.ci_vecs = self.ci_vecs.tocsr()

        # TODO: implement window search based on energy
        ovlp = np.abs((other.ci_vecs.tocsr() @ self.ci_vecs.getH()).toarray())
        mapped = ovlp.argmax(axis=1)

        # FIXME: return DataFrame
        return self._state_map[mapped], np.take_along_axis(ovlp, mapped[:, np.newaxis], axis=1).ravel()

    # def merge(self, other: 'MCStates', alignment, strategy: str = 'skip', ignore_space: bool = False) -> NoReturn:
    #     raise NotImplementedError('merge functionality is not implemented')
    #
    #     if not ignore_space and (self.mcspace is None or other.mcspace is None):
    #         warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')
    #
    #     if not ignore_space and self.mcspace != other.mcspace:
    #         raise ValueError('spaces of states are different, overlap is poorly defined')
    #
    #     for region in alignment:
    #         match region:
    #             case None, slice():
    #                 *_, sl = region
    #                 self.append(other[sl], ignore_space=ignore_space, reset_index=False)
    #
    #             case slice(), slice():
    #                 sl1, sl2 = region
    #                 if strategy == 'overwrite':
    #                     self[sl1] = other[sl2]
    #                 elif strategy == 'append':
    #                     self.append(other[sl2], ignore_space=ignore_space, reset_index=False)
    #
    #     self.sort(self.ENERGY_COL)

    # def append(self, other: 'MCStates', ignore_space: bool = False, reset_index: bool = True) -> NoReturn:
    #     """Extends the current MCStates with provided one.
    #
    #     The function does not check for duplicates or overlaps.
    #     """
    #     if not ignore_space and (self.mcspace is None or other.mcspace is None):
    #         warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')
    #
    #     if not ignore_space and self.mcspace != other.mcspace:
    #         raise ValueError('spaces of states are different, overlap is poorly defined')
    #
    #     if self.mcspace.n_act_mo != other.mcspace.n_act_mo:
    #         raise ValueError('number of active MOs is different between instances of MCStates, cannot transfer '
    #                          'rdm_diags')
    #
    #     warnings.warn('MCStates.extend() does not preserve computed properties')
    #
    #     # Extend rdm_diag
    #     self.rdm_diags = np.vstack((self.rdm_diags, other.rdm_diags))
    #
    #     # Extend ci_vec
    #     if sparse.isspmatrix_lil(self.ci_vecs):
    #         self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tolil()], format='lil', dtype=self.ci_vecs.dtype)
    #     else:
    #         self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tocsr()], format='csr', dtype=self.ci_vecs.dtype)
    #
    #     # Update state map
    #     new_state_map = np.arange(len(self) + len(other))
    #     new_state_map[:len(self)] = self._state_map + len(self)  # FIXME: might be a bug here
    #     self._state_map = new_state_map
    #
    #     # Extend df
    #     self._df = pd.concat([self._df[self.DEFAULT_COLS], other._df[self.DEFAULT_COLS]], axis=0, copy=True)
    #
    #     if self.peaks is not None or other.peaks is not None:
    #         warnings.warn('MCStates.extend() with peaks assigned is not tested')
    #
    #         # FIXME: implement peak extension
    #         if bool(self.peaks) != bool(other.peaks):
    #             valid_peaks = self.peaks if self.peaks is not None else other.peaks
    #             self.peaks = valid_peaks
    #         else:
    #             new_peaks_df = pd.concat(
    #                 [self.peaks.df[self.peaks.DEFAULT_COLS],
    #                  other.peaks.df[self.peaks.DEFAULT_COLS]],
    #                 axis=0, copy=True)
    #             self.peaks = self.peaks.__class__(new_peaks_df, states=self)
    #
    #     if reset_index:
    #         self.reset_index()

    def update_space(self, new_space: MCSpace, transform: ConfigTransform | None = None) -> NoReturn:
        # Update addresses
        addr_map = self.get_addr_map(new_space, transform)
        self.ci_vecs.resize(len(self), new_space.n_configs)
        self.ci_vecs.indices = addr_map.get(self.ci_vecs.indices).values.astype(self.ci_vecs.indices.dtype)

        # Remove old config label classes
        cols = [label for label in self.mcspace.config_class_labels if label in self._df]
        self._df.drop(columns=cols, inplace=True)

        # Update the space
        self.mcspace = new_space

    def get_addr_map(self, new_space: MCSpace, transform: ConfigTransform | None = None) -> pd.Series:
        addrs = np.unique(self.ci_vecs.indices)

        # Update configs
        configs = self.mcspace.graph.get_config(addrs)
        if transform is not None:
            transform(configs)

        new_addrs = new_space.graph.get_address(configs)
        return pd.Series(new_addrs, addrs)

    def get_state_ci_vec(self, state_idx: int | list[int], limit: int | None = None) -> pd.DataFrame:
        if isinstance(state_idx, int):
            state_idx = [state_idx]

        data = {
            self.IDX_NAME: np.empty(0, dtype=self.ci_vecs.indices.dtype),
            'addr': np.empty(0, dtype=self.ci_vecs.indices.dtype),
            'C': np.empty(0, dtype=self.ci_vecs.data.dtype),
            'norm': np.empty(0, dtype=np.float_),
        }

        for idx in state_idx:
            sl = np.s_[self.ci_vecs.indptr[idx]:self.ci_vecs.indptr[idx + 1]]
            addrs, coeffs = self.ci_vecs.indices[sl], self.ci_vecs.data[sl]

            coeffs_abs = np.abs(coeffs) ** 2
            sorted_idx = np.argsort(coeffs_abs)[::-1]
            selected = sorted_idx[:limit]

            data[self.IDX_NAME] = np.hstack((data[self.IDX_NAME], np.full_like(selected, fill_value=idx)))
            data['addr'] = np.hstack((data['addr'], addrs[selected]))
            data['C'] = np.hstack((data['C'], coeffs[selected]))
            data['norm'] = np.hstack((data['norm'], coeffs_abs[selected]))

        configs = self.mcspace.graph.get_config(data['addr'])
        if self.mcspace.graph.is_cas:
            data['config'] = configs[:, 1]
        else:
            data.update({f'r{i + 1}': configs[:, i] for i in range(self.mcspace.n_spaces)})

        data['config_repr'] = self.mcspace.graph.get_config_repr(configs)

        if len(self.mcspace.config_class_labels):
            addrs = np.unique(data['addr'])
            lookup = self.mcspace.get_address_class_lookup(addrs)
            data['config_class'] = np.vectorize(lambda i: lookup.get(i, ''))(data['addr'])

        df = pd.DataFrame(data)
        df.set_index(['idx', 'addr'], inplace=True)
        return df

    def __getitem__(self, key: int | slice) -> 'MCStates':
        """Implements simple slice indexing for MCStates.

        Notes:
            Indexing is done by integer position of state.
            Furthermore, the attributes of returned MCStates are views of parent object, as defined by numpy.
        """
        key = self._validate_key(key)

        new_df: pd.DataFrame = self._df.iloc[key]

        new_states = self.__class__(
            new_df.copy(deep=True),
            self.ci_vecs[self._state_map[key]],
            self.rdm_diags[self._state_map[key]],
            mcspace=self.mcspace,
        )

        if False and self.are_peaks_set:
            warnings.warn('Slicing MCStates with peaks assigned is not tested')

            preserve_key = new_df.index.values

            def cond(df):
                included_initial = df[self.transitions.INITIAL_COL].isin(preserve_key)
                included_final = df[self.transitions.FINAL_COL].isin(preserve_key)
                return included_initial & included_final

            self.transitions.calculate_state_idx()
            idx = self.transitions.filter(condition=cond)

            new_peaks_df = self.transitions.df[idx].copy(deep=True)
            new_states.transitions = self.transitions.__class__(new_peaks_df, states=new_states)

        return new_states

    def __setitem__(self, key: int | slice, other: 'MCStates') -> NoReturn:
        """Implements simple list-like indexing for MCStates.

        Notes:
            Indexing is done by integer position of state.
            Furthermore, the attributes of returned MCStates are views of parent object, as defined by numpy.
            Space equality is NOT checked, use at your own risk.
            Columns that are not defined in COLS are ignored in states.
            Columns that are not defined in COLS are cleared from self.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f'new_states must be an instance of {self.__class__.__name__}')

        if self.mcspace.n_mo_act != other.mcspace.n_mo_act:
            raise ValueError('Number of active MOs is different between instances of MCStates, cannot transfer '
                             'rdm_diags')

        # if self.space != states.space:
        #     raise ValueError('MCSpaces are not the same')

        if self.transitions is not None or other.transitions is not None:
            raise NotImplementedError('Index like setting is not implemented in the presence of peaks')

        key = self._validate_key(key)
        if (n := len(self._df.iloc[key])) != len(other):
            raise IndexError(f'trying to set {n} states, while len(new_states) = {len(other)}')

        # FIXME: don't clear calculated data
        self.clear_properties()
        self._df.iloc[key, self.DEFAULT_COLS] = other._df[self.DEFAULT_COLS]
        self._df.reset_index(drop=True, inplace=True)
        self._df.index.parser_class = self.IDX_NAME

        self.rdm_diags[self._state_map[key]] = other.rdm_diags

        found_lil = sparse.isspmatrix_lil(self.ci_vecs)
        ci_vec = self.ci_vecs if found_lil else self.ci_vecs.tolil()
        ci_vec[self._state_map[key], :] = other.ci_vecs.tolil()
        self.ci_vecs = ci_vec.tocsr() if found_lil else ci_vec

    def __delitem__(self, key: int | slice) -> NoReturn:
        key = self._validate_key(key)

        drop_key = self._df.index[key]
        if self.transitions is not None:
            warnings.warn('Slicing MCStates with peaks assigned is not tested')

            temp_df = self.transitions.calculate_state_idx()
            included_initial = temp_df[self.transitions.INITIAL_COL].isin(drop_key)
            included_final = temp_df[self.transitions.FINAL_COL].isin(drop_key)
            new_peaks_df = self.transitions._df[~(included_initial | included_final)].copy(deep=True)
            self.transitions = self.transitions.__class__(new_peaks_df, states=new_peaks_df)

        self._df.drop(drop_key, inplace=True)
        preserve_key = self._df.index.values  # indices of states that are preserved
        left_states = self._state_map[preserve_key]

        self._df.reset_index(drop=True, inplace=True)
        self._df.index.parser_class = self.IDX_NAME

        # TODO: try to use vstack
        found_lil = sparse.isspmatrix_lil(self.ci_vecs)
        ci_vec = self.ci_vecs if found_lil else self.ci_vecs.tolil()
        ci_vec = ci_vec[self._state_map[preserve_key]]
        self.ci_vecs = ci_vec.tocsr() if found_lil else ci_vec

        self.rdm_diags = np.delete(self.rdm_diags, self._state_map[key], axis=0)

        # Update state_map by finding inverse of sorting order
        self._state_map = np.empty_like(left_states)
        self._state_map[left_states.argsort()] = np.arange(left_states.shape[0])

    def _validate_key(self, key: int | slice) -> slice:
        if isinstance(key, int):
            if key < -len(self) or len(self) <= key:
                raise IndexError(f'index {key} is out of bounds for {self.__class__.__name__} '
                                 f'with {len(self)} states')

            key = slice(key, key + 1, 1)
        return key

    def print_state(self, include_mo: bool = True, include_config_class: bool = True,
                    config_limit: int = 5, shift_ground: bool = False,
                    idx: npt.ArrayLike | None = None, condition: Selector | None = None) -> NoReturn:
        BAR_WIDTH = 120

        idx = self.filter(idx=idx, condition=condition)

        E = self._df.loc[idx, self.ENERGY_COL] - (self.min_energy * shift_ground)

        mo_block_labels = [m for m in self.mcspace.mo_block_labels if m in self._df] if include_mo else []
        df_rdm = self._df.loc[idx, mo_block_labels].step_to(
            lambda r: ' + '.join([f'{l}({v:>5.2f})' for l, v in r.items()]), axis=1)

        config_class_labels = [c for c in self.mcspace.config_class_labels if
                               c in self._df] if include_config_class else []
        df_config_class = self._df.loc[idx, config_class_labels].step_to(
            lambda r: ' + '.join([f'{l}({v:7.3%})' for l, v in r.items()]), axis=1)

        print('=' * BAR_WIDTH)

        for i in idx:
            index = f'#{i:>5d}:'
            energy = f'E = {E.loc[i]:12.6G}Eh'
            print(index, energy)

            if pop_str := df_rdm.get(i):
                print(' ' * len(index) + ' MO: ' + pop_str)

            if ci_str := df_config_class.get(i):
                print(' ' * len(index) + ' |ψ> = ' + ci_str)

            print('-' * BAR_WIDTH)
            print(' Addr'.center(5), end=' ')
            if self.mcspace.graph.is_cas:
                print(f'CAS'.center(self.mcspace.graph.spaces.r2), end=' ')
            else:
                print(*[f'RAS{i + 1}'.center(w) for i, w in enumerate(self.mcspace.graph.spaces)], end=' ')
            print('Coefficient'.center(32))

            # FIXME: use get_state_ci_vec method here
            sl = np.s_[self.ci_vecs.indptr[i]:self.ci_vecs.indptr[i + 1]]
            addrs = self.ci_vecs.indices[sl]
            configs = self.mcspace.graph.get_config(addrs)
            coeffs = self.ci_vecs.data[sl]

            coeffs_abs = np.abs(coeffs) ** 2
            sorted_idx = np.argsort(coeffs_abs)[::-1]
            selected = sorted_idx[:config_limit]

            config_strs = self.mcspace.graph.get_config_repr(configs[selected])
            addrs = addrs[selected]
            coeffs = coeffs[selected]
            coeffs_abs = coeffs_abs[selected]
            for j in range(len(config_strs)):
                print(f'{addrs[j]:>6}',
                      config_strs[j],
                      f'C = {coeffs[j]:>13.3f}',
                      f'|C|^2 = {coeffs_abs[j]:>6.3f}')

            print('=' * BAR_WIDTH)

    @classmethod
    def from_dict(cls, data: ParsingResultType, /,
                  df_key: str = 'df_states',
                  ci_vecs_key: str = 'ci_vecs',
                  rdm_diags_key: str = 'rdm_diags',
                  source_key: str = 'source',
                  instance_key: str = 'states',
                  **kwargs) -> 'MCStates':
        if isinstance(instance := data.get(instance_key, None), cls):
            return instance
        elif isinstance(instance, dict):
            instance = cls.from_dict(instance, **kwargs)
        elif instance is None:
            data.update(kwargs)

            df: pd.DataFrame = data.pop(df_key)
            ci_vecs: sparse.coo_array = data.pop(ci_vecs_key)
            rdm_diags: npt.NDArray = data.pop(rdm_diags_key)

            try:
                space = MCSpace.from_dict(data)
                warnings.warn(f'Found {MCSpace}')
            except (KeyError, ValueError) as exc:
                warnings.warn(f'{MCSpace.__name__} is not found expect reduced '
                              f'functionality: {exc.args}', RuntimeWarning)
                space = None

            instance = data.setdefault(instance_key,
                                       cls(df, ci_vecs, rdm_diags,
                                           source=data.get(source_key, ''),
                                           mcspace=space,
                                           sort=kwargs.get('sort_states', False)))
            try:
                from .mctransitions import MCTransitions

                instance.transitions = MCTransitions.from_dict(data, states_key=instance_key, **kwargs)
            except KeyError:
                pass
        else:
            raise ValueError(f"{cls.__name__} did not recognized '{instance_key}' "
                             f"item in data: {instance}")

        return instance

    def validate_df(self: 'MCStates', new_df: pd.DataFrame) -> pd.DataFrame:
        new_df = super(MCStates, self).validate_df(new_df)

        if not (self.ci_vecs.ndim == self.rdm_diags.ndim == new_df.ndim == 2):
            raise ValueError("'ci_vec', 'rdm_diag', and 'df' must be 2D")

        if not (len(new_df.index) == self.ci_vecs.shape[0] == self.rdm_diags.shape[0]):
            raise ValueError("Number of states should match across 'df' DataFrame, "
                             "'ci_vecs' array, and 'rdm_diags' array")

        return new_df

    @property
    def is_space_set(self) -> bool:
        return self.mcspace is not None

    @property
    def are_peaks_set(self) -> bool:
        return self.transitions is not None

    @property
    def E(self) -> np.ndarray:
        return self._df[self.ENERGY_COL].values

    @property
    def min_energy(self) -> float:
        return float(self._df[self.ENERGY_COL].min())

    @property
    def max_energy(self) -> float:
        return float(self._df[self.ENERGY_COL].max())

    @property
    def energy_range(self) -> tuple[float, float]:
        return self.min_energy, self.max_energy

    def __repr__(self) -> str:
        emin, emax = self.energy_range
        energy_str = f'E=[{emin:>11.6f}, {emax:>11.6f}]'
        return f'{self.__class__.__name__}({energy_str}, #states={len(self)})'
