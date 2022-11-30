import json
from typing import NoReturn, Callable, IO, Any

import numpy as np
import pandas as pd
from scipy import sparse

from .cistring.graphs import RASGraph, RASMOs, Electrons
from .cistring.utils import ConfigArray, get_elec_count

__all__ = [
    'MCSpace',

    'MOBlock', 'MOBlocks',
    'ConfigClass', 'ConfigClasses',
    'ConfigTransform'
]


MOBlock = list[int | tuple[int, int]] | tuple[int, ...]
MOBlocks = dict[str, MOBlock]
ConfigClass = dict[str, int]
ConfigClasses = dict[str, ConfigClass]

ConfigTransform = Callable[[ConfigArray], NoReturn]


class MCSpace:
    """Active Space Definition and other info

    TODO: repackage mo_blocks and config_classes into a pd.DataFrame
    TODO: rewrite conversion using hdf5 format
    TODO: save config_class lookup
    """
    __slots__ = [
        'graph',
        '_mo_blocks', '_mo_block_labels', '_mo_masks',
        '_config_classes', '_config_class_labels'
    ]

    # mo: MOs  # (# inactive MOs, # active MOs, # Virtual MOs)
    # elec: Electrons
    graph: RASGraph

    _mo_blocks: MOBlocks | None
    _mo_block_labels: tuple[str, ...] | None
    _mo_masks: np.ndarray | None

    _config_classes: np.ndarray | None
    _config_class_labels: np.ndarray | None
    _config_class_lookup: pd.Series

    def __init__(self, graph: RASGraph, mo_blocks: MOBlocks | None = None, config_classes: ConfigClasses | None = None):
        self.graph = graph

        self._mo_blocks: MOBlocks | None = None
        self._mo_block_labels: tuple[str, ...] | None = None
        self._mo_masks: np.ndarray | None = None

        self._config_classes: np.ndarray | None = None
        self._config_class_labels: np.ndarray | None = None

        if mo_blocks is not None:
            self.mo_blocks = mo_blocks

        if config_classes is not None:
            self.config_classes = config_classes

    def partition_pdm_diag(self, pdm_diag: np.ndarray, mo_labels: list[str] | None = None) -> np.ndarray | pd.DataFrame:
        """Partitions PDM diagonal based on user defined spaces.

        Parameters:
            pdm_diag: np.ndarray -- must be of the dimensions (#States, #MO)
            mo_labels: Optional list of MOs to partition on. MO labels that are not in MO Blocks are ignored.
        """
        pdm_diag = np.asarray(pdm_diag).reshape(-1, self.n_act_mo)

        if mo_labels is None:
            mo_labels = self._mo_block_labels
        else:
            mo_labels = tuple(mo for mo in mo_labels if mo in self._mo_block_labels)

        pop = np.zeros((pdm_diag.shape[0], len(mo_labels)))
        for i, mo in enumerate(mo_labels):
            occ_list = self._mo_blocks[mo]
            pop[:, i] = pdm_diag[..., occ_list].sum(axis=1)

        return pd.DataFrame(pop, columns=mo_labels)

    def partition_ci_vec(self, ci_vec: sparse.csr_array,
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
        if self._config_classes is None:
            raise ValueError('Set config classes first.')

        occ = self.partition_config(config)
        assignments = (occ[..., np.newaxis, :] == self._config_classes).all(axis=-1)
        idx, categories = np.where(assignments)
        return idx, self._config_class_labels[categories]

    def partition_config(self, config: ConfigArray) -> np.ndarray:
        config = np.asarray(config, dtype=np.int64)  # .reshape(-1, self.n_ras)
        occ = get_elec_count(config[..., np.newaxis, :] & self._mo_masks)
        return occ.sum(axis=-1)

    @property
    def mo_blocks(self) -> MOBlocks:
        return self._mo_blocks

    @mo_blocks.setter
    def mo_blocks(self, mo_blocks: MOBlocks) -> NoReturn:
        self._mo_blocks = self.validate_mo_blocks(mo_blocks)
        self._mo_block_labels = tuple(self._mo_blocks)

        offsets = self.graph.get_mo_offsets()
        self._mo_masks = np.zeros((self.n_blocks, self.n_spaces), dtype=np.int64)
        for mo_block_idx, (label, occ_list) in enumerate(self._mo_blocks.items()):
            mask = self._mo_masks[mo_block_idx]
            for mo_idx in tuple(sorted(occ_list)):
                ras_idx = np.searchsorted(offsets, mo_idx, side='right') - 1
                pos = mo_idx - offsets[ras_idx]
                mask[ras_idx] |= (1 << pos)

        # self.assign_det_state.cache_clear()
        self._config_classes: np.ndarray | None = None
        self._config_class_labels: np.ndarray | None = None

    def validate_mo_blocks(self, mo_blocks: MOBlocks) -> MOBlocks:
        for label, mo_block in mo_blocks.items():
            # Expand compressed MO spaces as occupation lists
            occ_list: list[int] = []
            for mo_range in mo_block:
                match mo_range:
                    case [int(start), int(end)]:
                        assert 0 <= start <= end <= self.n_act_mo
                        occ_list.extend(range(start, end))
                    case int(index):
                        assert 0 <= index < self.n_act_mo
                        occ_list.append(index)
                    case _:
                        raise AssertionError(f'MO Space {mo_block} is invalid')

            mo_blocks[label] = tuple(occ_list)
        return mo_blocks

    @property
    def config_classes(self) -> ConfigClasses:
        det_categories = {}
        for idx, category in enumerate(self._config_classes):
            det_categories[self._config_class_labels[idx]] = {
                self._mo_block_labels[i]: int(cnt) for i, cnt in enumerate(category)
            }
        return det_categories

    @config_classes.setter
    def config_classes(self, config_classes: ConfigClasses) -> NoReturn:
        if not self._mo_blocks:
            raise ValueError("Set 'mo_blocks' first")

        config_classes = self.validate_config_classes(config_classes)

        self._config_class_labels = np.asarray(list(config_classes.keys()))
        n_config_classes = len(self._config_class_labels)

        self._config_classes = np.zeros((n_config_classes, self.n_blocks), dtype=np.int64)
        for idx, config_class in enumerate(config_classes.values()):
            self._config_classes[idx] = tuple(config_class.get(label, 0) for label in self._mo_block_labels)

    def validate_config_classes(self, config_classes: ConfigClasses) -> ConfigClasses:
        for class_label, config_class in config_classes.items():
            assert sum(config_class.values()) == self.n_elec_act

            for mo_space, mo_cnt in config_class.items():
                assert 0 <= mo_cnt
                assert mo_space in self.mo_blocks

        return config_classes

    @property
    def n_configs(self) -> int:
        return self.graph.n_configs

    @property
    def n_spaces(self) -> int:
        return self.graph.n_spaces

    @property
    def n_act_mo(self) -> int:
        return self.graph.n_mo

    @property
    def n_elec_act(self) -> int:
        return self.graph.n_elec

    @property
    def n_blocks(self) -> int:
        return len(self._mo_blocks)

    def to_dict(self) -> dict:
        return dict(
            ras=tuple(self.graph.spaces),
            elec=tuple(self.graph.elec),
            max_hole=self.graph.max_hole,
            max_elec=self.graph.max_elec,
            mo_blocks=self.mo_blocks,
            config_classes=self.config_classes
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
    def from_dict(cls, data: dict[str, Any], **kwargs) -> 'MCSpace':
        data.update(kwargs)

        return cls.from_space_spec(
            ras=data.pop('ras'),
            elec=data.pop('elec'),
            max_hole=data.pop('max_hole'),
            max_elec=data.pop('max_elec'),
            mo_blocks=data.pop('mo_blocks', None),
            config_classes=data.pop('config_classes', None),
        )

    @classmethod
    def from_space_spec(cls, ras: RASMOs | tuple[int, int, int] | list[int] | int,
                        elec: Electrons | tuple[int, int] | list[int] | int,
                        max_hole: int, max_elec: int, **kwargs) -> 'MCSpace':
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

        graph = RASGraph(ras, elec, max_hole, max_elec)
        return cls(graph, **kwargs)

    def __repr__(self) -> str:
        ras_spec = self.graph.get_graph_spec()
        return f'{self.__class__.__name__}([{ras_spec}])'

    def __eq__(self, other: 'MCSpace') -> bool:
        if self.graph != other.graph:
            return False

        if self._mo_blocks:
            if tuple(self._mo_blocks.items()) != tuple(other._mo_blocks.items()):
                return False

        if self._config_classes:
            if not (np.allclose(self._config_classes, other._config_classes) and
                    (self._config_class_labels == other._config_class_labels).all()):
                return False

        return True


def extend_ras1_factory(n: int) -> ConfigTransform:
    mask = (1 << n) - 1  # 1 bit at first n positions of an integer

    def extend_ras1(config: ConfigArray) -> ConfigArray:
        return (config << n) | mask

    return extend_ras1


if __name__ == '__main__':
    import os

    data_dir = os.path.join('data')

    space = MCSpace.from_space_spec(
        RASMOs(12, 14, 10),
        Electrons(13, 0),
        1, 1,
        mo_blocks={
            '3p': [(0, 12)],
            '4f': [(12, 26)],
            '5d': [(26, 36)],
        },
        config_classes={
            'f1': {'3p': 12, '4f': 1, '5d': 0},
            'f2': {'3p': 11, '4f': 2, '5d': 0},
            'd1': {'3p': 12, '4f': 0, '5d': 1},
            'f1d1': {'3p': 11, '4f': 1, '5d': 1},
        },
    )

    space.to_json(os.path.join(data_dir, 'rasci_1.space.json'))

    npz_archive = np.load(os.path.join(data_dir, 'example.npz'))
    pdm_diags = npz_archive['pdm_diags']
    df_pdm = space.partition_pdm_diag(pdm_diags)

    other = MCSpace.from_space_spec(
        RASMOs(6, 14, 10),
        Electrons(7, 0),
        1, 1,
        mo_blocks={
            '3p': [(0, 6)],
            '4f': [(6, 20)],
            '5d': [(20, 30)],
        },
        config_classes={
            'f1': {'3p': 6, '4f': 1, '5d': 0},
            'f2': {'3p': 5, '4f': 2, '5d': 0},
            'd1': {'3p': 6, '4f': 0, '5d': 1},
            'f1d1': {'3p': 5, '4f': 1, '5d': 1},
        },
    )

    other.to_json(os.path.join(data_dir, 'rasci_2.space.json'))
