import warnings

from typing import NoReturn, Callable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .mcpeaks import MCPeaks
    from .mcspace import MCSpace, ConfigTransform

import numpy as np
import numpy.typing as npt
import pandas as pd

from scipy import sparse

from .base import MCBase


__all__ = [
    'MCStates',
    'Selector',
]


Selector = Callable[[pd.DataFrame], bool]


class MCStates(MCBase):
    """Holds information about multiconfigurational states.

    Attributes:
        space: CAS/RAS definition, and related utilities.
        df: Energy (in Eh) of states, and any other information that can be associated
            with a particular state. States are indexed internally by df.index starting from 0, with unique index
            defined by pair of (STATE_COL, SOURCE_COL). df.shape = (#States, #Properties)
        ci_vec: CI vectors as in Sparse Compressed Row format. ci_vec.shape = (#States, #Determinants)
        rdm_diag: Diagonals of 1RDM matrices for every state. rdm_diag.shape = (#States, #Active MOs)

    Possible columns on df:
        SOURCE_COL: file from which the state originates;
        STATE_COL: index of the state as defined in the `source`;
        E_COL: Energy of the state in Hartree;

        IDX_COLS: list of columns used to uniquely identify states;
        DEFAULT_COLS: list of permanent columns on `df` for MCStates object to be valid;

    Future Development:
        TODO: add name attribute
        TODO: Explore LIL sparse structure for manipulation of ci_vec
        TODO: pd.MultiIndex for permanent cols, MO blocks, config classes, and other computed properties
        TODO: Adapt to using dense representation of CI vectors
        TODO: try using STATE_COL and SOURCE_COL as index in df
        TODO: include duplicate and similar
        TODO: Move _state_map to RESOURCE_COL
    """
    IDX_NAME = 'idx'

    E_COL = 'E'
    DEGENERACY_COL = 'g'
    STATE_COL = 'state'
    SOURCE_COL = 'state_source'
    RESOURCE_COL = 'resource_idx'  # index of related resource in ci_vecs and rdm_diag, see _state_map

    DEFAULT_COLS = [STATE_COL, SOURCE_COL, E_COL]  # Permanent property columns for the df
    IDX_COLS = [STATE_COL, SOURCE_COL]  # Columns used to identify the states uniquely

    ci_vecs: sparse.csr_array | sparse.lil_array  # Sparse array of CI vectors
    rdm_diags: np.ndarray

    space: MCSpace | None
    peaks: MCPeaks | None

    # Used for implicit sorting of rdm_diag and ci_vec
    # Move to df
    _state_map: np.ndarray

    def __init__(self,
                 df: pd.DataFrame,
                 ci_vecs: sparse.csr_array | sparse.lil_array | sparse.coo_array,
                 rdm_diags: np.ndarray, /,
                 source: str = '',
                 space: MCSpace | None = None, *,
                 sort: bool = False) -> NoReturn:
        if self.SOURCE_COL not in df:
            if source:
                df[self.SOURCE_COL] = source
            else:
                raise ValueError(f"either 'df' must have {self.SOURCE_COL} or "
                                 f"source argument must be passed to {self.__class__.__name__}")

        if not (ci_vecs.ndim == rdm_diags.ndim == df.ndim == 2):
            raise ValueError("'ci_vec', 'rdm_diag', and 'df' must be 2D")

        if not (len(df.index) == ci_vecs.shape[0] == rdm_diags.shape[0]):
            raise ValueError("Number of states should match across 'df' DataFrame, "
                             "'ci_vecs' array, and 'rdm_diags' array")

        self.ci_vecs = ci_vecs.tocsr()
        self.rdm_diags = rdm_diags
        self.df = df
        self.reset_index()

        if space is not None:
            if space.n_act_mo != rdm_diags.shape[1]:
                raise ValueError("Number of active MOs in MCSpace must be equal to number of MOs in 'rdm_diags' array")

            if space.n_configs != ci_vecs.shape[1]:
                raise ValueError("Number of configurations in MCSpace must be equal to number of determinants in "
                                 "'ci_vecs' array")

        self.space = space
        self._state_map = np.arange(len(self))
        self.peaks = None
        # self.df['resource_idx'] = self._state_map

        if sort:
            self.sort(self.E_COL)

    def sort(self, col: str = E_COL) -> NoReturn:
        idx = np.argsort(self.df_[col].values)
        self._state_map = self._state_map[idx]
        self.df_ = self.df_.iloc[idx]

        self.reset_index()
        if self.peaks is not None:
            if self.peaks.INITIAL_COL in self.peaks.df or self.peaks.FINAL_COL in self.peaks.df:
                self.peaks.calculate_state_idx(save=True, replace=True)

    def analyze(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                save=True, replace=False, **kwargs) -> pd.DataFrame | None:
        """Performs analysis of states, using the following functions:
            - estimate_state_degeneracy
            - calculate_ci_vec_norm
            - partition_rdm_diag
            - partition_ci_vec

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
            self.calculate_ci_vec_norm(**kwargs, idx=idx, condition=condition, save=save, replace=replace),
        ]

        try:
            dfs.append(self.partition_rdm_diag(idx=idx, condition=condition, save=save, replace=replace))
            dfs.append(self.partition_ci_vec(idx=idx, condition=condition, save=save, replace=replace))
        except ValueError as err:
            warnings.warn(err.args[0])

        if not save:
            return pd.concat([self.df_[self.E_COL], *dfs], axis=1)

        if self.peaks is not None:
            self.peaks.analyze(save=True)

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
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)
        df = self.space.partition_rdm_diag(self.rdm_diags[self._state_map[idx]])
        df.set_index(self.df_.index[idx], inplace=True)

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
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)
        df = self.space.partition_ci_vec(self.ci_vecs[self._state_map[idx]])
        df.set_index(self.df_.index[idx], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_ci_vec_norm(self, idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                              save: bool = True, replace: bool = False, col_name: str = 'norm', **kwargs):
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
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, condition=condition)
        norm = sparse.linalg.norm(self.ci_vecs, axis=1)
        df = pd.DataFrame({col_name: norm}, index=self.df_.index[idx])

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def estimate_state_degeneracy(self, tol: float = 1e-3,
                                  idx: npt.ArrayLike | None = None, condition: Selector | None = None,
                                  save: bool = True, replace: bool = True,
                                  col_name: str = DEGENERACY_COL) -> pd.DataFrame | None:
        """

        Args:
            tol:
            idx:
            condition:
            save:
            replace:
            col_name:

        Returns:

        """
        idx = self.filter(idx=idx, condition=condition)

        dE = self.df_.loc[self.df_.index[idx], self.E_COL].diff()
        dE.fillna(0, inplace=True)
        df = pd.DataFrame({col_name: (dE > tol).cumsum()})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def find_similar(self, other: 'MCStates', /, ignore_space: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not ignore_space and (self.space is None or other.space is None):
            warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')

        if not ignore_space and self.space != other.space:
            raise ValueError('spaces of states are different, overlap is poorly defined')

        self.ci_vecs = self.ci_vecs.tocsr()

        # TODO: implement window search based on energy
        ovlp = np.abs((other.ci_vecs.tocsr() @ self.ci_vecs.getH()).toarray())
        mapped = ovlp.argmax(axis=1)

        # FIXME: return DataFrame
        return self._state_map[mapped], np.take_along_axis(ovlp, mapped[:, np.newaxis], axis=1).ravel()

    def merge(self, other: 'MCStates', alignment, strategy: str = 'skip', ignore_space: bool = False) -> NoReturn:
        raise NotImplementedError('merge functionality is not implemented')

        if not ignore_space and (self.space is None or other.space is None):
            warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')

        if not ignore_space and self.space != other.space:
            raise ValueError('spaces of states are different, overlap is poorly defined')

        for region in alignment:
            match region:
                case None, slice():
                    *_, sl = region
                    self.extend(other[sl], ignore_space=ignore_space, reset_index=False)

                case slice(), slice():
                    sl1, sl2 = region
                    if strategy == 'overwrite':
                        self[sl1] = other[sl2]
                    elif strategy == 'append':
                        self.extend(other[sl2], ignore_space=ignore_space, reset_index=False)

        self.sort(self.E_COL)

    def extend(self, other: 'MCStates', ignore_space: bool = False, reset_index: bool = True) -> NoReturn:
        """Extends the current MCStates with provided one.

        The function does not check for duplicates or overlaps.
        """
        if not ignore_space and (self.space is None or other.space is None):
            warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')

        if not ignore_space and self.space != other.space:
            raise ValueError('spaces of states are different, overlap is poorly defined')

        if self.space.n_act_mo != other.space.n_act_mo:
            raise ValueError('number of active MOs is different between instances of MCStates, cannot transfer '
                             'rdm_diags')

        warnings.warn('MCStates.extend() does not preserve computed properties')

        # Extend rdm_diag
        self.rdm_diags = np.vstack((self.rdm_diags, other.rdm_diags))

        # Extend ci_vec
        if sparse.isspmatrix_lil(self.ci_vecs):
            self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tolil()], format='lil', dtype=self.ci_vecs.dtype)
        else:
            self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tocsr()], format='csr', dtype=self.ci_vecs.dtype)

        # Update state map
        new_state_map = np.arange(len(self) + len(other))
        new_state_map[:len(self)] = self._state_map + len(self)  # FIXME: might be a bug here
        self._state_map = new_state_map

        # Extend df
        self.df_ = pd.concat([self.df_[self.DEFAULT_COLS], other.df_[self.DEFAULT_COLS]], axis=0, copy=True)

        if self.peaks is not None or other.peaks is not None:
            warnings.warn('MCStates.extend() with peaks assigned is not tested')

            # FIXME: implement peak extension
            if bool(self.peaks) != bool(other.peaks):
                valid_peaks = self.peaks if self.peaks is not None else other.peaks
                self.peaks = valid_peaks
            else:
                new_peaks_df = pd.concat(
                    [self.peaks.df[self.peaks.DEFAULT_COLS],
                     other.peaks.df[self.peaks.DEFAULT_COLS]],
                    axis=0, copy=True)
                self.peaks = self.peaks.__class__(new_peaks_df, states=self)

        if reset_index:
            self.reset_index()

    def update_space(self, new_space: MCSpace, transform: ConfigTransform | None = None) -> NoReturn:
        if transform is not None:
            addr_map = self.get_addr_map(new_space, transform)

            # Update addresses
            self.ci_vecs.resize(len(self), new_space.n_configs)
            self.ci_vecs.indices = addr_map.get(self.ci_vecs.indices).values

        # Remove old config label classes
        cols = [label for label in self.space.config_class_labels if label in self.df_]
        self.df_.drop(columns=cols, inplace=True)

        # Update the space
        self.space = new_space

    def get_addr_map(self, new_space: MCSpace, transform: ConfigTransform | None) -> pd.Series:
        addrs = np.unique(self.ci_vecs.indices)

        # Update configs
        configs = self.space.graph.get_config(addrs)
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

        configs = self.space.graph.get_config(data['addr'])
        if self.space.graph.is_cas:
            data['config'] = configs[:, 1]
        else:
            data.update({f'r{i + 1}': configs[:, i] for i in range(self.space.n_spaces)})

        data['config_repr'] = self.space.graph.get_config_repr(configs)

        if len(self.space.config_class_labels):
            addrs = np.unique(data['addr'])
            lookup = self.space.get_address_class_lookup(addrs)
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

        new_df: pd.DataFrame = self.df_.iloc[key]

        new_states = self.__class__(
            new_df.copy(deep=True),
            self.ci_vecs[self._state_map[key]],
            self.rdm_diags[self._state_map[key]],
            space=self.space,
        )

        if self.peaks is not None:
            warnings.warn('Slicing MCStates with peaks assigned is not tested')

            preserve_key = new_df.index.values
            temp_df = self.peaks.calculate_state_idx()
            included_initial = temp_df[self.peaks.INITIAL_COL].isin(preserve_key)
            included_final = temp_df[self.peaks.FINAL_COL].isin(preserve_key)
            new_peaks_df = self.peaks.df_[included_initial & included_final].copy(deep=True)
            new_states.peaks = self.peaks.__class__(new_peaks_df, states=new_states)

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

        if self.space.n_act_mo != other.space.n_act_mo:
            raise ValueError('Number of active MOs is different between instances of MCStates, cannot transfer '
                             'rdm_diags')

        # if self.space != states.space:
        #     raise ValueError('MCSpaces are not the same')

        if self.peaks is not None or other.peaks is not None:
            raise NotImplementedError('Index like setting is not implemented in the presence of peaks')

        key = self._validate_key(key)
        if (n := len(self.df_.iloc[key])) != len(other):
            raise IndexError(f'trying to set {n} states, while len(new_states) = {len(other)}')

        # FIXME: don't clear calculated data
        self.clear_properties()
        self.df_.iloc[key, self.DEFAULT_COLS] = other.df_[self.DEFAULT_COLS]
        self.df_.reset_index(drop=True, inplace=True)
        self.df_.index.name = self.IDX_NAME

        self.rdm_diags[self._state_map[key]] = other.rdm_diags

        found_lil = sparse.isspmatrix_lil(self.ci_vecs)
        ci_vec = self.ci_vecs if found_lil else self.ci_vecs.tolil()
        ci_vec[self._state_map[key], :] = other.ci_vecs.tolil()
        self.ci_vecs = ci_vec.tocsr() if found_lil else ci_vec

    def __delitem__(self, key: int | slice) -> NoReturn:
        key = self._validate_key(key)

        drop_key = self.df_.index[key]
        if self.peaks is not None:
            warnings.warn('Slicing MCStates with peaks assigned is not tested')

            temp_df = self.peaks.calculate_state_idx()
            included_initial = temp_df[self.peaks.INITIAL_COL].isin(drop_key)
            included_final = temp_df[self.peaks.FINAL_COL].isin(drop_key)
            new_peaks_df = self.peaks.df_[~(included_initial | included_final)].copy(deep=True)
            self.peaks = self.peaks.__class__(new_peaks_df, states=new_peaks_df)

        self.df_.drop(drop_key, inplace=True)
        preserve_key = self.df_.index.values  # indices of states that are preserved
        left_states = self._state_map[preserve_key]

        self.df_.reset_index(drop=True, inplace=True)
        self.df_.index.name = self.IDX_NAME

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

        E = self.df_.loc[idx, self.E_COL] - (self.min_energy * shift_ground)

        mo_block_labels = [m for m in self.space.mo_block_labels if m in self.df_] if include_mo else []
        df_rdm = self.df_.loc[idx, mo_block_labels].apply(
            lambda r: ' + '.join([f'{l}({v:>5.2f})' for l, v in r.items()]), axis=1)

        config_class_labels = [c for c in self.space.config_class_labels if c in self.df_] if include_config_class else []
        df_config_class = self.df_.loc[idx, config_class_labels].apply(
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
            print(
                ' Addr'.center(5),
                *[f'RAS{i + 1}'.center(w) for i, w in enumerate(self.space.graph.spaces)],
                'Coefficient'.center(32)
            )

            # FIXME: use get_state_ci_vec method here
            sl = np.s_[self.ci_vecs.indptr[i]:self.ci_vecs.indptr[i + 1]]
            addrs = self.ci_vecs.indices[sl]
            configs = self.space.graph.get_config(addrs)
            coeffs = self.ci_vecs.data[sl]

            coeffs_abs = np.abs(coeffs) ** 2
            sorted_idx = np.argsort(coeffs_abs)[::-1]
            selected = sorted_idx[:config_limit]

            config_strs = self.space.graph.get_config_repr(configs[selected])
            addrs = addrs[selected]
            coeffs = coeffs[selected]
            coeffs_abs = coeffs_abs[selected]
            for j in range(len(config_strs)):
                print(f'{addrs[j]:>6}',
                      config_strs[j],
                      f'C = {coeffs[j]:>13.3f}',
                      f'|C|^2 = {coeffs_abs[j]:+5.3f}')

            print('=' * BAR_WIDTH)

    @classmethod
    def from_dict(cls, data: dict[str, Any], /,
                  df_key: str = 'df_states',
                  ci_vecs_key: str = 'ci_vecs',
                  rdm_diags_key: str = 'rdm_diags',
                  space_key: str = 'space',
                  source_key: str = 'source',
                  **kwargs) -> 'MCStates':
        data.update(kwargs)

        df = data.pop(df_key)
        ci_vecs = data.pop(ci_vecs_key)
        rdm_diags = data.pop(rdm_diags_key)

        space = data.pop(space_key, None)
        if isinstance(space, dict):
            from .mcspace import MCSpace
            space = MCSpace.from_dict(space)

        states = cls(
            df, ci_vecs, rdm_diags,
            source=data.get(source_key, ''), space=space,
            sort=kwargs.get('sort_states', False)
        )

        try:
            from .mcpeaks import MCPeaks

            states.peaks = MCPeaks.from_dict(data, states=states, states_key='states')
        except KeyError:
            pass

        return states

    @property
    def df(self) -> pd.DataFrame:
        return self.df_

    @df.setter
    def df(self, new_df: pd.DataFrame) -> NoReturn:
        for col in self.DEFAULT_COLS:
            if col not in new_df:
                raise ValueError(f"'df' must have {col}")

        if not (self.ci_vecs.ndim == self.rdm_diags.ndim == new_df.ndim == 2):
            raise ValueError('df must be 2D')

        self.df_ = new_df

    @property
    def E(self) -> np.ndarray:
        return self.df_[self.E_COL].values

    @property
    def min_energy(self) -> np.float64:
        return np.float64(self.df_[self.E_COL].min())

    @property
    def max_energy(self) -> np.float64:
        return np.float64(self.df_[self.E_COL].max())

    @property
    def energy_range(self) -> tuple[np.float64, np.float64]:
        return self.min_energy, self.max_energy

    def __repr__(self) -> str:
        emin, emax = self.energy_range
        energy_str = f'E=[{emin:>11.6f}, {emax:>11.6f}]'
        return f'{self.__class__.__name__}({energy_str}, #states={len(self)})'


if __name__ == '__main__':
    import os, copy

    from mctools.parser.gaussian.utils import parse_gdvlog
    from mctools.parser.gaussian.l910 import l910_parser_funcs_general

    data_dir = os.path.join('data')
    space1_filename = os.path.join(data_dir, 'rasci_1.space.json')
    gdvlog1 = os.path.join(data_dir, 'rasci_1.log')

    space1 = MCSpace.from_json(space1_filename)
    data1 = parse_gdvlog(gdvlog1, l910_parser_funcs_general, n_ground=14)
    states1 = MCStates.from_dict(data1, space=space1)

