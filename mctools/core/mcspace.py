from __future__ import annotations

import json
import warnings

from typing import NoReturn, Callable, IO, Literal, Sequence, ClassVar, Type, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

from .cistring.graphs import RASGraph, RASMOs, Electrons
from .cistring.utils import ConfigArray, get_elec_count


if TYPE_CHECKING:
    from parsing.core.pattern import ParsingResultType

__all__ = [
    'MOSpacePartition',
    'MCSpace',
    'ConfigTransform',
]

ConfigTransform = Callable[[ConfigArray], NoReturn]


class MOSpacePartition:
    SpacesTuple: ClassVar[Type] = tuple[int, int, int, int, int]

    __slots__ = [
        'spaces',
        'mcspace',

        'offsets',
    ]

    spaces: SpacesTuple
    mcspace: MCSpace | None

    offsets: SpacesTuple

    def __init__(self, spaces: SpacesTuple | tuple[int, int, int] | tuple[int], *, mcspace: MCSpace | None = None) -> None:
        if any(o < 0 for o in spaces):
            raise ValueError(f'Number of orbitals in spaces must be non-negative: {spaces!r}')

        if len(spaces) == 1:
            self.spaces = (0, 0, *spaces, 0, 0)
        elif len(spaces) == 3:
            self.spaces = (0, *spaces, 0)
        elif len(spaces) == 5:
            self.spaces = spaces
        else:
            raise ValueError(f'Number of spaces must be equal to 5: {spaces!r}: '
                             f'frozen core, inactive, active, secondary, frozen virtual')

        self.mcspace = mcspace
        if self.mcspace is not None and mcspace.n_mo_act != self.n_active:
            raise ValueError(f'Number of active spaces in MOSpaces must be equal '
                             f'to the number of active spaces in MCSpace: {self.n_active} != {self.mcspace.n_mo_act}')

        offsets = [0]
        for n_o in self.spaces[:-1]:
            offsets.append(offsets[-1] + n_o)
        self.offsets = tuple(offsets)

    @property
    def frozen_core(self) -> slice:
        return np.s_[0:self.offsets[1]]

    @property
    def frozen_core_2d(self) -> tuple[slice, ...]:
        return np.s_[0:self.offsets[1], 0:self.offsets[1]]

    @property
    def inactive(self) -> slice:
        return np.s_[self.offsets[1]:self.offsets[2]]

    @property
    def inactive_2d(self) -> tuple[slice, ...]:
        return np.s_[self.offsets[1]:self.offsets[2], self.offsets[1]:self.offsets[2]]

    @property
    def active(self) -> slice:
        return np.s_[self.offsets[2]:self.offsets[3]]

    @property
    def active_2d(self) -> tuple[slice, ...]:
        return np.s_[self.offsets[2]:self.offsets[3], self.offsets[2]:self.offsets[3]]

    @property
    def secondary(self) -> slice:
        return np.s_[self.offsets[3]:self.offsets[4]]

    @property
    def secondary_2d(self) -> tuple[slice, ...]:
        return np.s_[self.offsets[3]:self.offsets[4], self.offsets[3]:self.offsets[4]]

    @property
    def frozen_virtual(self) -> slice:
        return np.s_[self.offsets[4]:self.offsets[5]]

    @property
    def frozen_virtual_2d(self) -> tuple[slice, ...]:
        return np.s_[self.offsets[4]:self.offsets[5], self.offsets[4]:self.offsets[5]]

    @property
    def n_frozen_core(self) -> int:
        return self.spaces[0]

    @property
    def n_core(self) -> int:
        return self.spaces[1]

    @property
    def n_inactive(self) -> int:
        return self.n_frozen_core + self.n_core

    @property
    def n_active(self) -> int:
        return self.spaces[2]

    @property
    def n_secondary(self) -> int:
        return self.n_virtual + self.n_frozen_virtual

    @property
    def n_virtual(self) -> int:
        return self.spaces[3]

    @property
    def n_frozen_virtual(self) -> int:
        return self.spaces[4]

    @property
    def n(self) -> int:
        return sum(self.spaces)

    @classmethod
    def from_spaces(cls, active: int, /, core: int = 0,
                    virtual: int = 0, frozen_core: int = 0,
                    frozen_virtual: int = 0) -> 'MOSpacePartition':
        spaces = (frozen_core, core, active, virtual, frozen_virtual)
        return cls(spaces)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(' \
               f'Inactive=({self.n_frozen_core},{self.n_inactive}), ' \
               f'Active={self.n_active}, ' \
               f'Secondary=({self.n_virtual},{self.n_frozen_virtual}))'


class MCSpace:
    """Active Space Definition and other info

    Attributes:
        graph: RASGraph instance --- provides CI string manipulation routines
        df: pd.DataFrame --- stores definitions of MOBlocks and ConfigClasses

    TODO: rewrite file saving using hdf5 format
    TODO: save config_class lookup
    """
    __slots__ = [
        'graph',
        'df',
    ]

    # Useful typing definitions
    MOBlock = Sequence[int | tuple[int, int] | tuple[int, ...]] | tuple[int, ...]
    RawMOBlocks = dict[str, MOBlock]
    MOBlocks = pd.Series

    ConfigClass = dict[str, int]
    ConfigClasses = dict[str, ConfigClass]

    graph: RASGraph
    df: pd.DataFrame | None

    MO_OCC_COL = 'mo_occ'
    MO_MASK_COL = 'mo_mask'
    MO_ORB_COL = 'mo_orb'
    MO_N_ORB_COL = 'mo_n_orb'
    MO_N_ELEC_COL = 'mo_n_elec'
    MO_BLOCK_IDX_NAME = 'block'

    CONFIG_CLASS_COL = 'config_class'

    def __init__(self, graph: RASGraph, /,
                 df: pd.DataFrame | None = None, *,
                 mo_blocks: RawMOBlocks | None = None,
                 config_classes: ConfigClasses | None = None):
        self.graph = graph
        self.df = df

        if df is None and mo_blocks is not None:
            self.set_mo_blocks(mo_blocks)

        if config_classes is not None:
            self.set_config_classes(config_classes)

    def partition_rdm_diag(self, rdm_diag: np.ndarray, /, mo_blocks: list[str] | None = None) -> pd.DataFrame:
        """Partitions RDM diagonal based on user defined spaces.

        Parameters:
            rdm_diag: np.ndarray -- must be of the dimensions (#States, #MO)
            mo_blocks: Optional list of MOs to partition on. MO labels that are not in MO Blocks are ignored.
        """
        if not self.are_mo_blocks_set:
            raise ValueError('Set MO Blocks first.')

        rdm_diag = np.asarray(rdm_diag)

        if mo_blocks is None:
            occ_list = self.df[self.MO_OCC_COL]
        else:
            occ_list = self.df.loc[mo_blocks, self.MO_OCC_COL]

        return occ_list.dot(rdm_diag.T).T

    def partition_ci_vec(self, ci_vec: sparse.csr_array, /,
                         coefficient_col: str = 'C', state_col: str = 'state',
                         address_col: str = 'addr', config_class_col: str = 'config_class',
                         unmapped_class: str = 'rest') -> pd.DataFrame:
        addrs = np.unique(ci_vec.indices)
        lookup = self.get_address_class_lookup(addrs)

        df = pd.DataFrame({
            coefficient_col: np.abs(ci_vec.data) ** 2,
            state_col: np.digitize(np.arange(ci_vec.data.shape[0]), ci_vec.indptr) - 1,
            address_col: ci_vec.indices.astype(np.uint64),
        })

        df[config_class_col] = df[address_col].map(lookup).fillna(unmapped_class)

        result = df.pivot_table(
            index=state_col, columns=config_class_col, values=coefficient_col,
            aggfunc=np.sum, fill_value=0
        )

        return result

    def get_address_class_lookup(self, addrs: np.ndarray) -> pd.Series:
        configs = self.graph.get_config(addrs)
        assigned, labels = self.get_config_class(configs)
        return pd.Series(data=labels, index=addrs[assigned].astype(np.uint64))

    def get_config_class(self, config: ConfigArray) -> tuple[np.ndarray, np.ndarray]:
        if not self.are_config_classes_set:
            raise ValueError('Set config classes first.')

        occ = self.partition_config(config)
        config_classes = self.df[self.CONFIG_CLASS_COL].dropna(axis=0).astype(np.int_).values.T
        assignments = (occ[..., np.newaxis, :] == config_classes).all(axis=-1)

        idx, categories = np.where(assignments)
        config_class_labels = np.asarray(self.df[self.CONFIG_CLASS_COL].columns, dtype=np.str_)
        return idx, config_class_labels[categories]

    def partition_config(self, config: ConfigArray) -> np.ndarray:
        if not self.are_config_classes_set:
            raise ValueError('Set config classes first.')

        config = np.asarray(config, dtype=self.graph.config_dtype).reshape(-1, self.n_spaces)
        config_classes = self.df.loc[:, [self.CONFIG_CLASS_COL, self.MO_MASK_COL]].dropna(axis=0)
        masks = config_classes.loc[:, self.MO_MASK_COL].values
        occ = get_elec_count(config[..., np.newaxis, :] & masks, config_dtype=self.graph.config_dtype)
        return occ.sum(axis=-1)

    def set_mo_blocks(self, raw_blocks: RawMOBlocks, /, replace: bool = True) -> NoReturn:
        blocks = self.validate_mo_blocks(raw_blocks)
        new_df = self.transform_mo_blocks(blocks)

        if self.df is None or replace:
            self.df = new_df
        else:
            # Update rows that are already present on the self.df
            to_update = new_df.index.intersection(self.df.index)
            self.df.loc[to_update].update(new_df)

            # Update rows that are already present on the self.df
            to_concat = new_df.index.difference(self.df.index)
            self.df = pd.concat([self.df, new_df.loc[to_concat]], axis=0, copy=True)

        self.df.sort_index(axis=1, inplace=True)

    def validate_mo_blocks(self, blocks: RawMOBlocks, /) -> MOBlocks:
        """Validates MO blocks and transforms them into pd.Series.

        Assumes that they are provided in the type defined by self.MOBlocks.

        Args:
            blocks: blocks to be validated. Blocks should be unique.

        Raises:
            ValueError: if blocks are invalid

        Returns:
            pd.Series object with index
        """
        for label, block in blocks.items():
            # Expand compressed MO spaces as occupation lists
            occ_list: list[int] = []
            for mo_range in block:
                match mo_range:
                    case [int(start), int(end)]:
                        if not (0 <= start <= end <= self.n_mo_act):
                            raise ValueError(f'MO range {mo_range} in {label} block is invalid: '
                                             f'indices must be within [0, {self.n_mo_act}]')
                        occ_list.extend(range(start, end))

                    case int(index):
                        if not (0 <= index < self.n_mo_act):
                            raise ValueError(f'MO range {mo_range} in {label} block is invalid: '
                                             f'indices must be within [0, {self.n_mo_act}]')
                        occ_list.append(index)

                    case [*indices]:
                        if not all(0 <= i < self.n_mo_act for i in indices):
                            raise ValueError(f'MO range {mo_range} in {label} block is invalid: '
                                             f'indices must be within [0, {self.n_mo_act}]')
                        occ_list.extend(indices)

                    case _:
                        raise ValueError(f'MO range {mo_range} in {label} block is invalid:'
                                         f'indices must be integers or tuples of ints')

            blocks[label] = tuple(sorted(occ_list))

        s = pd.Series(blocks, name=self.MO_ORB_COL)
        s.index.name = self.MO_BLOCK_IDX_NAME
        return s

    def transform_mo_blocks(self, blocks: MOBlocks, /) -> pd.DataFrame:
        df_occ = self.get_mo_block_occ(blocks)
        df_mask = self.get_mo_block_masks(blocks)

        df_orb = blocks.to_frame()
        df_orb.columns = pd.MultiIndex.from_tuples([(self.MO_ORB_COL, '')])

        df = pd.concat([df_occ, df_mask, df_orb], axis=1)

        # Calculate number of orbitals per each block
        df[self.MO_N_ORB_COL] = df[self.MO_OCC_COL].sum(axis=1)

        # Calculate number of electrons per each block
        mo_idx = np.arange(self.n_mo_act)
        df[self.MO_N_ELEC_COL] = df[self.MO_OCC_COL].step_to(lambda x: (mo_idx < self.n_elec_act)[x].sum(), axis=1)

        return df

    def get_mo_block_occ(self, blocks: MOBlocks, /, dtype=np.bool_) -> pd.DataFrame:
        """Transform blocks into occupation array.

        Parameters:
            blocks: Assumes that blocks are presented in expanded form and have been validated
        """
        occ = np.full((len(blocks), self.n_mo_act), fill_value=False, dtype=dtype)

        for i, occ_list in enumerate(blocks.values):
            occ[i, occ_list] = True

        cols = pd.MultiIndex.from_product(([self.MO_OCC_COL], np.arange(self.n_mo_act)))
        return pd.DataFrame(occ, index=blocks.index, columns=cols)

    def get_mo_block_masks(self, blocks: MOBlocks, /) -> pd.DataFrame:
        """Generates mask for each block that can be applied to configurations to get population of respective block.

        Masks can span across multiple spaces.

        Args:
            blocks: Assumes that blocks are presented in expanded form and have been validated.
            dtype: should be an integer type that can fit an entire space. np.int64 is recommended.

        Returns:
            Array of shape (#blocks, #spaces), where each row corresponds to MO block and column corresponds to
            a space, as defined in graph.
        """
        masks = np.zeros((len(blocks), self.n_spaces), dtype=self.graph.config_dtype)

        offsets = self.graph.get_mo_offsets()
        for block_idx, mo_orbs in enumerate(blocks.values):
            mask = masks[block_idx]
            for mo_idx in mo_orbs:
                ras_idx = np.searchsorted(offsets, mo_idx, side='right') - 1
                pos = int(mo_idx - offsets[ras_idx])
                mask[ras_idx] |= (1 << pos)

        cols = pd.MultiIndex.from_product(([self.MO_MASK_COL], np.arange(self.n_spaces)))
        return pd.DataFrame(masks, index=blocks.index, columns=cols)

    def set_config_classes(self, raw_classes: ConfigClasses, /,
                           definition: Literal['occupation', 'change'] = 'occupation',
                           replace=True) -> NoReturn:
        if not self.are_mo_blocks_set:
            raise ValueError('Set MO Blocks first')

        df = self.validate_config_classes(raw_classes, definition=definition)

        if replace:
            self.df.drop(self.CONFIG_CLASS_COL, axis=1, inplace=True, errors='ignore')
            self.df = pd.concat([self.df, df], axis=1, copy=True)
        else:
            old = self.df.loc[:, (self.CONFIG_CLASS_COL, slice(None))].columns
            new = df.loc[:, (self.CONFIG_CLASS_COL, slice(None))].columns

            # Update rows that are already present on the self.df
            to_update = new.intersection(old)
            self.df.loc[:, to_update].update(df.loc[:, to_update], errors='raise')

            # Update rows that are already present on the self.df
            to_concat = new.difference(old)
            self.df = pd.concat([self.df, df.loc[:, to_concat]], axis=1, copy=True)

    def validate_config_classes(self, classes: ConfigClasses, /,
                                definition: Literal['occupation', 'change'] = 'occupation',
                                defaults: bool = True) -> pd.DataFrame:
        """Validates Configuration Classes and transforms them into occupation based definition"""
        # Convert to dataframe, blocks that are not defined in self.df.index are ignored
        if defaults:
            df = pd.DataFrame(classes, index=self.df.index).transpose()
        else:
            raise NotImplementedError('Non-default behaviour for config classes is not yet implemented.')

        df = df.astype(np.float_, errors='raise')  # Convert to float, if raise error on failure

        if definition == 'change':
            df.fillna(0, downcast='infer', inplace=True)
            df += df[self.MO_N_ELEC_COL]
        elif definition != 'occupation':
            raise ValueError("invalid 'definition': can only be 'occupation' or 'change'")

        df.fillna(self.df[self.MO_N_ELEC_COL], downcast='infer', inplace=True)
        if (invalid_idx := df.sum(axis=1) != self.n_elec_act).any():
            warnings.warn('Provided classes have invalid MO blocks, dropping them...', RuntimeWarning)
            df.drop(df.index[invalid_idx], axis=0, inplace=True)

        if len(df.index) == 0:
            warnings.warn('No valid config classes found', RuntimeWarning)

        df = df.transpose()
        df.columns = pd.MultiIndex.from_product(([self.CONFIG_CLASS_COL], df.columns))
        return df

    @property
    def mo_blocks(self) -> RawMOBlocks:
        return self.df[self.MO_ORB_COL].to_dict()

    @mo_blocks.setter
    def mo_blocks(self, mo_blocks: RawMOBlocks) -> NoReturn:
        self.set_mo_blocks(mo_blocks, replace=True)

    @property
    def are_mo_blocks_set(self) -> bool:
        return self.df is not None and len(self.df.index) > 0

    @property
    def config_classes(self) -> ConfigClasses:
        if self.CONFIG_CLASS_COL in self.df:
            return self.df[self.CONFIG_CLASS_COL].to_dict()
        return {}

    @config_classes.setter
    def config_classes(self, config_classes: ConfigClasses) -> NoReturn:
        self.set_config_classes(config_classes, definition='occupation', replace=True)

    @property
    def are_config_classes_set(self) -> bool:
        return self.are_mo_blocks_set and \
               self.CONFIG_CLASS_COL in self.df and \
               len(self.df[self.CONFIG_CLASS_COL].columns) > 0

    @property
    def n_configs(self) -> int:
        return self.graph.n_configs

    @property
    def n_spaces(self) -> int:
        return self.graph.n_spaces

    @property
    def n_mo_act(self) -> int:
        return self.graph.n_mo

    @property
    def n_elec_act(self) -> int:
        return self.graph.n_elec

    @property
    def n_blocks(self) -> int:
        return len(self.df.index)

    @property
    def mo_block_labels(self) -> np.ndarray:
        if self.df is None:
            labels = []
        else:
            labels = self.df.index
        return np.asarray(labels, dtype=np.str_)

    @property
    def config_class_labels(self) -> np.ndarray:
        if self.df is None:
            labels = []
        else:
            labels = self.df[self.CONFIG_CLASS_COL].columns if self.CONFIG_CLASS_COL in self.df else []
        return np.asarray(labels, dtype=np.str_)

    def to_dict(self) -> dict:
        return dict(
            ras=tuple(self.graph.spaces),
            elec=tuple(self.graph.elec),
            max_hole=self.graph.max_hole,
            max_elec=self.graph.max_elec,
            mo_blocks=self.mo_blocks,
            config_classes=self.config_classes,
        )

    def to_json(self, fp: str | IO, **kwargs) -> NoReturn:
        data = self.to_dict()

        match fp:
            case str(filename):
                with open(filename, 'w') as f:
                    json.dump(data, f, **kwargs)
            case IO(fp):
                json.dump(data, fp, **kwargs)
            case _:
                raise ValueError('Invalid fp')

    @classmethod
    def from_json(cls, fp: str | IO, **kwargs) -> 'MCSpace':
        match fp:
            case str(filename):
                with open(filename, 'r') as f:
                    data = json.load(f, **kwargs)
            case IO(fp):
                data = json.load(fp, **kwargs)
            case _:
                raise ValueError('Invalid fp')

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: ParsingResultType, /,
                  active_spaces_key: str = 'active_spaces',
                  active_elec_key: str = 'elec_act',
                  max_hole_key: str = 'max_hole',
                  max_elec_key: str = 'max_elec',
                  instance_key: str = 'mcspace',
                  **kwargs) -> 'MCSpace':
        if isinstance(instance := data.get(instance_key, None), cls):
            return instance
        elif isinstance(instance, dict):
            instance = cls.from_dict(instance, **kwargs)
        elif instance is None:
            data.update(kwargs)

            instance = data.setdefault(
                instance_key, cls.from_space_spec(
                    data.pop(active_spaces_key),
                    data.pop(active_elec_key),
                    data.pop(max_hole_key), data.pop(max_elec_key),
                    mo_blocks=data.pop('mo_blocks', None),
                    config_classes=data.pop('config_classes', None),
                    use_python_int=data.pop('use_python_int', None),
                )
            )
        return instance

    @classmethod
    def from_space_spec(cls, ras: RASMOs | tuple[int, int, int] | list[int] | int,
                        elec: Electrons | tuple[int, int] | list[int] | int, /,
                        max_hole: int = 0, max_elec: int = 0,  **kwargs) -> 'MCSpace':
        match ras:
            case int(cas):
                ras = RASMOs(0, cas, 0)
            case int(r1), int(r2), int(r3):
                ras = RASMOs(r1, r2, r3)
            case _:
                raise ValueError('Incorrect RAS MOs')

        match elec:
            case int(e):
                elec = Electrons(e, 0)
            case int(e_a), int(e_b):
                elec = Electrons(e_a, e_b)
            case _:
                raise ValueError('Incorrect Electrons')

        graph_kwargs = {k: kwargs.pop(k)
                        for k in ['reverse', 'cat_order', 'use_python_int']
                        if k in kwargs}

        graph = RASGraph(ras, elec, max_hole, max_elec, **graph_kwargs)
        return cls(graph, **kwargs)

    def __repr__(self) -> str:
        ras_spec = self.graph.get_graph_spec()
        return f'{self.__class__.__name__}([{ras_spec}], #Det={self.n_configs:,d}, #Cat={self.graph.n_cat:,d})'

    def __eq__(self, other: 'MCSpace') -> bool:
        return self.graph == other.graph  # and self.df.equals(other.df)


def extend_ras1_factory(n: int) -> ConfigTransform:
    mask = (1 << n) - 1  # 1 bit at first n positions of an integer

    def extend_ras1(config: ConfigArray) -> ConfigArray:
        return (config << n) | mask

    return extend_ras1
