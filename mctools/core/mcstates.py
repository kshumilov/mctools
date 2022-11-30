import warnings
from typing import NoReturn, Callable, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import sparse

from .mcspace import MCSpace, ConfigTransform


__all__ = [
    'MCStates',
    'Selector'
]


Selector = Callable[[pd.DataFrame], bool]


class MCStates:
    """Holds information about multiconfigurational states.

    Attributes:
        space: CAS/RAS definition, and related utilities.
        df: Energy (in Eh) of states, and any other information that can be associated
            with a particular state. Indexed by state id starting from 0. df.shape = (#States, #Properties).
            df must contain the following columns:
                - E: np.float64 --- Energy of states in Eh;
                - origin: str --- name of the file from which the state is coming from;
                - idx: np.uint32 --- original index of the state in the filename;
        ci_vec: CI vectors as in Sparse Compressed Row format. ci_vec.shape = (#States, #Determinants)
        pdm_diag: Diagonals of 1PDM matrices for every state. pdm_diag.shape = (#States, #Active MOs)

    TODO: add name attribute
    TODO: Explore LIL sparse structure for manipulation of ci_vec
    TODO: pd.MultiIndex for permanent cols, MO blocks, config classes, and other computed properties
    TODO: rewrite assertions as Errors
    """
    __slots__ = [
        'df',
        'ci_vecs',
        'pdm_diags',
        'space',
        '_state_map'
    ]

    IDX_NAME = 'idx'

    # Permanent cols
    E_COL = 'E'
    STATE_COL = 'state'
    SOURCE_COL = 'source'

    COLS = [E_COL, STATE_COL, SOURCE_COL]
    IDX_COLS = [STATE_COL, SOURCE_COL]

    df: pd.DataFrame
    ci_vecs: sparse.csr_array | sparse.lil_array
    pdm_diags: np.ndarray

    space: MCSpace | None

    # Used for implicit sorting of pdm_diag and ci_vec
    _state_map: np.ndarray

    # Columns on the df that have been computed
    # _mo_block_labels: set[str] = field(repr=False, default_factory=set)
    # _config_class_labels: set[str] = field(repr=False, default_factory=set)

    def __init__(self,
                 states: pd.DataFrame,
                 ci_vecs: sparse.csr_array | sparse.lil_array | sparse.coo_array,
                 pdm_diag: np.ndarray,
                 source: str = '',
                 space: MCSpace | None = None,
                 sort_states: bool = False):

        if not (ci_vecs.ndim == pdm_diag.ndim == states.ndim == 2):
            raise ValueError('ci_vec, pdm_diag, and states must be 2D')

        if self.E_COL not in states:
            raise ValueError(f'states must have {self.E_COL}')

        if self.STATE_COL not in states:
            raise ValueError(f'states must have {self.STATE_COL}')

        if self.SOURCE_COL not in states:
            if source:
                states[self.SOURCE_COL] = source
            else:
                raise ValueError(f'either states must have {self.SOURCE_COL} or source argument must be passed to '
                                 f'{self.__class__.__name__}')

        assert len(states.index) == ci_vecs.shape[0] == pdm_diag.shape[0]

        self.df = states
        self.reset_index()

        self.ci_vecs = ci_vecs.tocsr()
        self.pdm_diags = pdm_diag

        if space is not None:
            assert space.n_act_mo == pdm_diag.shape[1]
            assert space.n_configs == ci_vecs.shape[1]

        self.space = space
        self._state_map = np.arange(len(self))

        if sort_states:
            self.sort(self.E_COL)

    def sort(self, col: str = E_COL) -> NoReturn:
        idx = np.argsort(self.df[col].values)
        self._state_map = self._state_map[idx]
        self.df = self.df.iloc[idx]

        self.reset_index()

    def filter(self, idx: npt.ArrayLike | None = None, cond: Selector | None = None,
               label_index: bool = False) -> np.ndarray | pd.Index:
        """Filter states on positional integer index and some condition.

        States are selected based on provided index first, and then based on condition. If no idx or condition is
        provided, the function returns array of indices to all states.

        TODO: return MCStates object
        """
        idx = np.asarray(idx).reshape(-1) if idx is not None else np.arange(len(self))
        selected = cond(self.df.iloc[idx]) if cond else np.s_[...]

        if label_index:
            return self.df.index[selected]

        return idx[selected]

    def analyze(self, idx: npt.ArrayLike | None = None, cond: Selector | None = None,
                save: bool = True, replace: bool = False, **kwargs) -> pd.DataFrame | None:
        self.clear_properties()

        dfs = [
            self.estimate_state_degeneracy(**kwargs, idx=idx, cond=cond, save=save, replace=replace),
            self.calculate_ci_vec_norm(**kwargs, idx=idx, cond=cond, save=save, replace=replace),
        ]

        try:
            dfs.append(self.partition_pdm_diag(idx=idx, cond=cond, save=save, replace=replace))
            dfs.append(self.partition_ci_vec(idx=idx, cond=cond, save=save, replace=replace))
        except ValueError as err:
            warnings.warn(err.args[0])

        if not save:
            return pd.concat([self.df[self.E_COL], *dfs], axis=1)

    def partition_pdm_diag(self, idx: npt.ArrayLike | None = None, cond: Selector | None = None,
                           save: bool = True, replace: bool = False, **kwargs) -> pd.DataFrame | None:
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, cond=cond)
        df = self.space.partition_pdm_diag(self.pdm_diags[self._state_map[idx]])
        df.set_index(self.df.index[idx], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def partition_ci_vec(self, idx: npt.ArrayLike | None = None, cond: Selector | None = None,
                         save: bool = True, replace: bool = False, **kwargs) -> pd.DataFrame | None:
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, cond=cond)
        df = self.space.partition_ci_vec(self.ci_vecs[self._state_map[idx]])
        df.set_index(self.df.index[idx], inplace=True)

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def calculate_ci_vec_norm(self, idx: npt.ArrayLike | None = None, cond: Selector | None = None,
                              save: bool = True, replace: bool = False, **kwargs):
        if self.space is None:
            raise ValueError('Set MCSpace first')

        idx = self.filter(idx=idx, cond=cond)
        norm = sparse.linalg.norm(self.ci_vecs, axis=1)
        df = pd.DataFrame({'norm': norm}, index=self.df.index[idx])

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def estimate_state_degeneracy(self, tol: float = 1e-3,
                                  idx: npt.ArrayLike | None = None, cond: Selector | None = None,
                                  save: bool = True, replace: bool = True) -> pd.DataFrame | None:
        idx = self.filter(idx=idx, cond=cond)

        dE = self.df.loc[self.df.index[idx], self.E_COL].diff()
        dE.fillna(0, inplace=True)
        df = pd.DataFrame({'g': (dE > tol).cumsum()})

        if not save:
            return df

        self.update_properties(df, replace=replace)

    def find_similar(self, other: 'MCStates', ignore_space: bool = False) -> tuple[np.ndarray, np.ndarray]:
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
        """Extends the current MCStates with provided once."""
        if not ignore_space and (self.space is None or other.space is None):
            warnings.warn('At least one of MCStates does not have well defined MCSpace, proceed with caution')

        if not ignore_space and self.space != other.space:
            raise ValueError('spaces of states are different, overlap is poorly defined')

        if self.space.n_act_mo != other.space.n_act_mo:
            raise ValueError('number of active MOs is different between instances of MCStates, cannot transfer '
                             'pdm_diags')

        # Extend pdm_diag
        self.pdm_diags = np.vstack((self.pdm_diags, other.pdm_diags))

        # Extend ci_vec
        if sparse.isspmatrix_lil(self.ci_vecs):
            self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tolil()], format='lil', dtype=self.ci_vecs.dtype)
        else:
            self.ci_vecs = sparse.vstack([self.ci_vecs, other.ci_vecs.tocsr()], format='csr', dtype=self.ci_vecs.dtype)

        # Update state map
        new_state_map = np.arange(len(self) + len(other))
        new_state_map[:len(self)] = self._state_map
        self._state_map = new_state_map

        # Extend df
        self.df = pd.concat([self.df[self.COLS], other.df[self.COLS]], axis=0, copy=True)

        if reset_index:
            self.reset_index()

    def update_space(self, new_space: MCSpace, transform: ConfigTransform | None = None) -> NoReturn:
        if transform is not None:
            addr_map = self.get_addr_map(new_space, transform)

            # Update addresses
            self.ci_vecs.resize(len(self), new_space.n_configs)
            self.ci_vecs.indices = addr_map.get(self.ci_vecs.indices).values

        # Remove old config label classes
        cols = [l for l in self.space._config_class_labels if l in self.df]
        self.df.drop(columns=cols, inplace=True)

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
            data.update({f'r{i + 1}': configs[:, i] for i in range(self.space.n_ras)})

        data['config_repr'] = self.space.graph.get_config_repr(configs)

        if self.space._config_class_labels is not None:
            addrs = np.unique(data['addr'])
            lookup = self.space.get_address_class_lookup(addrs)
            data['config_class'] = np.vectorize(lambda i: lookup.get(i, ''))(data['addr'])

        df = pd.DataFrame(data)
        df.set_index(['idx', 'addr'], inplace=True)
        return df

    def clear_properties(self) -> NoReturn:
        curr_cols = set(self.df.columns) - set(self.COLS)
        self.df.drop(columns=curr_cols, inplace=True)

    def update_properties(self, new_df: pd.DataFrame, replace: bool = False) -> NoReturn:
        cols = set(new_df.columns)
        curr_cols = set(self.df.columns) - set(self.COLS)
        previous_cols = cols & curr_cols

        if replace:
            self.df.drop(columns=previous_cols, inplace=True)
            previous_cols.clear()

        new_cols = cols - previous_cols
        self.df = pd.concat([self.df, new_df[list(new_cols)]], axis=1, copy=False)
        self.df.update(new_df[list(previous_cols)])

    def reset_index(self) -> NoReturn:
        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = self.IDX_NAME

    def __getitem__(self, key: int | slice) -> 'MCStates':
        """Implements simple slice indexing for MCStates.

        Note: Indexing is done by integer position of state.
        Furthermore, the attributes of returned MCStates are views of parent object, as defined by numpy.
        """
        key = self._validate_key(key)

        return self.__class__(
            self.df.iloc[key],
            self.ci_vecs[self._state_map[key]],
            self.pdm_diags[self._state_map[key]],
            space=self.space,
        )

    def __setitem__(self, key: int | slice, other: 'MCStates') -> NoReturn:
        """Implements simple list-like indexing for MCStates.

        Note: Indexing is done by integer position of state.
        Furthermore, the attributes of returned MCStates are views of parent object, as defined by numpy.
        Space equality is NOT checked, use at your own risk.
        Columns that are not defined in COLS are ignored in states.
        Columns that are not defined in COLS are cleared from self.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(f'new_states must be an instance of {self.__class__.__name__}')

        if self.space.n_act_mo != other.space.n_act_mo:
            raise ValueError('Number of active MOs is different between instances of MCStates, cannot transfer '
                             'pdm_diags')

        # if self.space != states.space:
        #     raise ValueError('MCSpaces are not the same')

        key = self._validate_key(key)
        if (n := len(self.df.iloc[key])) != len(other):
            raise ValueError(f'trying to set {n} states, while len(new_states) = {len(other)}')

        # FIXME: don't clear calculated data
        self.clear_properties()
        self.df.iloc[key, self.COLS] = other.df[self.COLS]
        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = self.IDX_NAME

        self.pdm_diags[self._state_map[key]] = other.pdm_diags

        found_lil = sparse.isspmatrix_lil(self.ci_vecs)
        ci_vec = self.ci_vecs if found_lil else self.ci_vecs.tolil()
        ci_vec[self._state_map[key], :] = other.ci_vecs.tolil()
        self.ci_vecs = ci_vec.tocsr() if found_lil else ci_vec

    def __delitem__(self, key: int | slice) -> NoReturn:
        key = self._validate_key(key)
        self.df.drop(self.df.index[key], inplace=True)
        preserve_key = self.df.index.values  # indices of states that are preserved
        left_states = self._state_map[preserve_key]

        self.df.reset_index(drop=True, inplace=True)
        self.df.index.name = self.IDX_NAME

        # TODO: try to use vstack
        found_lil = sparse.isspmatrix_lil(self.ci_vecs)
        ci_vec = self.ci_vecs if found_lil else self.ci_vecs.tolil()
        ci_vec = ci_vec[self._state_map[preserve_key]]
        self.ci_vecs = ci_vec.tocsr() if found_lil else ci_vec

        self.pdm_diags = np.delete(self.pdm_diags, self._state_map[key], axis=0)

        # Update state_map by finding inverse of sorting order
        self._state_map = np.empty_like(left_states)
        self._state_map[left_states.argsort()] = np.arange(left_states.shape[0])

    def _validate_key(self, key: int | slice) -> slice:
        if isinstance(key, int):
            if key >= len(self) or key < -len(self):
                raise IndexError(f'index {key} is out of bounds for {self.__class__.__name__} '
                                 f'with {len(self)} states')

            key = slice(key, key + 1, 1)
        return key

    def print_state(self, include_mo: bool = False, include_config_class: bool = False,
                    config_limit: int = 5, shift_ground: bool = False,
                    idx: npt.ArrayLike | None = None,
                    cond: Selector | None = None) -> NoReturn:
        idx = self.filter(idx=idx, cond=cond)

        E = self.df.loc[idx, self.E_COL] - (self.min_energy and shift_ground)

        mo_block_labels = [m for m in self.space._mo_block_labels if m in self.df] if include_mo else []
        df_pdm = self.df.loc[idx, mo_block_labels].apply(
            lambda r: '+'.join([f'{l}({v:>5.2f})' for l, v in r.items()]), axis=1)

        config_class_labels = [c for c in self.space._config_class_labels if c in self.df] if include_config_class else []
        df_config_class = self.df.loc[idx, config_class_labels].apply(
            lambda r: '+'.join([f'{l}({v:7.3%})' for l, v in r.items()]), axis=1)

        print('=' * 79)

        for i in idx:
            index = f'#{i:>5d}:'
            energy = f'E = {E.loc[i]:12.6e}Eh'
            header = [index, energy]

            if pop_str := df_pdm.get(i):
                header.append('MO: ' + pop_str)

            if ci_str := df_config_class.get(i):
                header.append('|ψ> = ' + ci_str)

            print(*header)
            print('-' * 79)
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

            print('=' * 79)

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs) -> 'MCStates':
        data.update(kwargs)

        states = data.pop('df_states')
        ci_vecs = data.pop('ci_vecs')
        pdm_diags = data.pop('pdm_diags')

        space = data.pop('space', None)
        if isinstance(space, dict):
            space = MCSpace.from_dict(space)

        return cls(
            states, ci_vecs, pdm_diags,
            source=data.get('source'), space=space,
            sort_states=kwargs.get('sort_states', False)
        )

    @property
    def E(self) -> np.ndarray:
        return self.df[self.E_COL].values

    @property
    def min_energy(self) -> np.float64:
        return self.df[self.E_COL].min()

    @property
    def max_energy(self) -> np.float64:
        return self.df[self.E_COL].max()

    @property
    def energy_range(self) -> tuple[np.float64, np.float64]:
        return self.min_energy, self.max_energy

    def __repr__(self) -> str:
        emin, emax = self.energy_range
        energy_str = f'E=[{emin:>11.6f}, {emax:>11.6f}]'
        return f'{self.__class__.__name__}({energy_str}, #states={len(self)})'

    def __len__(self) -> int:
        return len(self.df)


if __name__ == '__main__':
    import os, copy

    from mctools.parser.gaussian.utils import parse_gdvlog
    from mctools.parser.gaussian.l910 import l910_parser_funcs

    data_dir = os.path.join('data')
    space1_filename = os.path.join(data_dir, 'rasci_1.space.json')
    gdvlog1 = os.path.join(data_dir, 'rasci_1.log')

    space1 = MCSpace.from_json(space1_filename)
    data1 = parse_gdvlog(gdvlog1, l910_parser_funcs, n_ground=14)
    states1 = MCStates.from_dict(data1, space=space1)

    space2_filename = os.path.join(data_dir, 'rasci_2.space.json')
    gdvlog2 = os.path.join(data_dir, 'rasci_2.log')

    space2 = MCSpace.from_json(space2_filename)
    data2 = parse_gdvlog(gdvlog2, l910_parser_funcs, n_ground=14)
    states2 = MCStates.from_dict(data2, space=space2)

    mask = (((1 << 4) - 1) << 8) | 3

    def transform(configs):
        configs[..., 0] <<= 2
        configs[..., 0] |= mask

    states2.update_space(space1, transform=transform)

    # df_example = pd.read_csv(os.path.join(data_dir, 'states.csv'), index_col='state')
    # ci_vec_example = sparse.csr_array(sparse.load_npz(os.path.join(data_dir, 'ci_vecs_small.npz')), copy=True)
    # pdm_diags_example = data['pdm_diags']
    #
    # states_example = MCStates(df_example, ci_vec_example, pdm_diags_example, space=space_example)
    # states_example.analyze()
    #
    # s1 = copy.deepcopy(states_example[:14])
    # s2 = copy.deepcopy(states_example[20:30])
    # s3 = copy.deepcopy(s1)
    # s4 = copy.deepcopy(states_example[:40])
    # s5 = copy.deepcopy(states_example[25:45])
    #
    # s1.extend(s2)
    # s1.extend(s4)

    # s1.sort(s1.E_COL)
    #
    # del s1[:2]
    #
    # print(s1.df)
    # print(align_states(s1, s5))
