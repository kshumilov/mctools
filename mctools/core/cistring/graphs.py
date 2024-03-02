from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import NoReturn, Literal, ClassVar

import attr
import attrs
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.special import comb

from mctools.core.cistring.utils import *
from mctools.newcore.consolidator import Archived
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY

__all__ = [
    'SimpleGraph',
    'DASGraph',
]


@attr.define(repr=False, eq=False)
class SimpleGraph:
    """Basic CI string manipulations such as addressing, excitation lists, etc.

    In this implementation, the unoccupied edges of the graph are taken to be zero.
    This means that only occupied orbitals play roles in formation of the CI configuration address.

    Attributes:
        n_orb: int --- # Number of orbitals in the space
        n_elec: int --- # Number of electrons in the space

    TODO: Rewrite nodes and edges formation in rectangle format (i.e. transform parallelogram graph into rectangle)
    TODO: implement <I|J>, p^+|J>, p|J>, excitation lists, etc...
    TODO: Fix dtypes and typing
    TODO: Use dtype with multiple int fields to manipulate strings
    """
    BYTE_TO_BITS: ClassVar[int] = 8

    n_orb: int = attrs.field(
        default=0,
        converter=int,
        validator=attr.validators.ge(0),
    )

    n_elec: int = attrs.field(
        default=0,
        converter=int,
        validator=attr.validators.ge(0),
    )

    @n_elec.validator
    def _validate_n_elec(self, attribute: attrs.Attribute, n_elec: int):
        if not (0 <= n_elec <= self.n_orb):
            raise ValueError(f'Invalid definition of active space: ({n_elec}e|{self.n_orb}o)')

    reverse: bool = False

    config_dtype: npt.DTypeLike = attrs.field()

    @config_dtype.default
    def _get_default_config_dtype(self) -> npt.DTypeLike:
        if self.n_orb > 64:
            return np.dtype(object)
        return np.dtype(np.int64)

    max_norb: float | int = attrs.field()

    @max_norb.default
    def _get_default_max_norb(self) -> int | float:
        if self.config_dtype.type is object:
            return np.inf
        return self.config_dtype.itemsize * self.BYTE_TO_BITS

    edges: npt.NDArray[np.uint64] = attrs.field(default=None, init=False)
    offsets: npt.NDArray[np.uint64] = attrs.field(default=None, init=False)

    def __attrs_post_init__(self) -> None:
        if self.n_orb > self.max_norb:
            raise ValueError(f'{self.__class__.__name__} is unable to support space with {self.n_orb} orbitals. '
                             f'Current maximum is {self.max_norb} orbitals.')

        nodes = self.get_nodes()
        self.edges = self.get_edges(nodes)
        self.offsets = self.get_offsets(self.edges)

    @property
    def n_configs(self) -> int:
        return get_num_configs(self.n_orb, self.n_elec)

    @property
    def space_tuple(self) -> tuple[int, int]:
        return self.n_orb, self.n_elec

    def get_edges(self, nodes: npt.NDArray[np.uint64], /) -> npt.NDArray[np.uint64]:
        Y = np.zeros((self.n_elec, self.n_orb), dtype=np.uint64)

        e_idx, o_idx = self.get_node_idx()
        if self.reverse:
            for e in range(self.n_elec):
                o = o_idx[e_idx == e]
                Y[e, o] = nodes[e + 1, o]

        else:
            for e in range(self.n_elec):
                o = o_idx[e_idx == e]
                Y[e, o[1:]] = np.cumsum(nodes[e + 1, o[1:] + 1])

        return Y

    def get_nodes(self) -> npt.NDArray[np.uint64]:
        """Builds weights of nodes in graph according to the given ordering.

        In 'reverse' ordering paths run (0, 0) -> (e, o), while in 'direct' (#e, #o) -> (e, o).
        The weight of a node is thus can be written as:
            nodes[e, o] = (len(path) choose #diagonals steps)
                        = (o choose e) <--> reverse
                        OR
                        = (#o - o choose #e - e) <--> direct
        """
        nodes = np.zeros((self.n_elec + 1, self.n_orb + 1), dtype=np.uint64)

        if self.reverse:
            f = lambda e, o: comb(o, e, exact=True)
        else:
            f = lambda e, o: comb(self.n_orb - o, self.n_elec - e, exact=True)

        f = np.vectorize(f, otypes=[np.uint64])

        idx = self.get_node_idx()
        nodes[idx] = f(*idx)

        return nodes

    def get_node_idx(self) -> tuple[np.ndarray, np.ndarray]:
        """Builds indices of non-zero elements of the graph.

        For convenience the Full CI graph is stored in (#elec + 1, #orb + 1)-shaped arrays.
        However, there are only (#elec + 1) * (#orb - #elec + 1) non-zero elements
        which form a slanted array.

        For examples, here is directly ordered weights of CAS(7o, 3e) in Kramer's unrestricted formalism:
              <--#orb - #elec + 1-->
              0   1   2   3   4   5   6   7 --- #orb + 1
           ---------------------------------
        0 -|  1   1   1   1   1   0   0   0
        1  |  0   1   2   3   4   5   0   0
        2  |  0   0   1   3   6   10  15  0
        3  |  0   0   0   1   4   10  20  35
        |
        #elec + 1

        Returns:
            Tuple of arrays, where the first array are electrons indices and second — orbitals.
        """
        l_o = self.n_orb - self.n_elec
        e_idx, o_idx = np.indices((self.n_elec + 1, l_o + 1), dtype=np.uint64)
        o_idx += e_idx
        return e_idx.ravel(), o_idx.ravel()

    def get_offsets(self, edges: npt.NDArray[np.uint64], /) -> npt.NDArray[np.uint64]:
        offsets = edges.copy()
        for e in range(self.n_elec - 2, -1, -1):
            offsets[e] += np.roll(offsets[e + 1], -1)
        return offsets

    def get_address(self, config: ConfigArray) -> AddrArray:
        config = np.asarray(config, dtype=self.config_dtype)
        config &= self.space_mask

        addr = np.zeros_like(config, dtype=np.uint64)

        o = 0
        e = np.zeros_like(config)
        bit = np.zeros_like(addr, dtype=self.config_dtype)

        while (idx := e < self.n_elec).any():
            # bit = ((config[idx] >> o) & ONE
            np.right_shift(config, o, out=bit, where=idx, dtype=self.config_dtype, casting='unsafe')
            np.bitwise_and(bit, 1, out=bit, where=idx, dtype=self.config_dtype)

            # e[idx] += bit
            np.add(e, bit, out=e, where=idx, dtype=e.dtype, casting='unsafe')

            # update address with edges
            addr[idx] += (self.edges[e[idx] - 1, o] * bit[idx]).astype(addr.dtype)

            o += 1
        return addr

    def get_config(self, addr: np.ndarray) -> np.ndarray:
        addr = np.asarray(addr, dtype=np.uint64).copy()
        config = np.zeros_like(addr, dtype=self.config_dtype)

        e = 0
        e_idx, o_idx = self.get_node_idx()
        while e < self.n_elec:
            o = o_idx[e_idx == e]
            j = np.searchsorted(self.offsets[e, o], addr, side='right') - 1
            addr -= self.edges[e, o[j]]
            config |= np.left_shift(1, o[j], dtype=self.config_dtype)
            e += 1
        return config

    def get_all_configs(self) -> np.ndarray:
        if self.n_elec == 0:
            return np.asarray([0], dtype=self.config_dtype)

        if self.reverse:
            configs = get_configs_reverse(self.n_orb, self.n_elec)
        else:
            configs = get_configs_direct(self.n_orb, self.n_elec)

        return np.asarray(configs, dtype=self.config_dtype)

    def get_config_repr(self, config: ConfigArray) -> npt.NDArray[np.str_]:
        return get_config_repr(config, (self.n_orb,), config_dtype=self.config_dtype)

    @property
    def space_mask(self) -> int | np.int64:
        raw_mask = 2 ** self.n_orb - 1
        return self.config_dtype.type(raw_mask)

    def get_graph_spec(self) -> str:
        return f'[{self.n_orb:>2d}o, {self.n_elec:>2d}e]'

    def __repr__(self) -> str:
        spec = self.get_graph_spec()
        return f'{self.__class__.__name__}({spec}, #Det={self.n_configs:,d})'

    def print_weights(self) -> NoReturn:
        # TODO: correctly implement printing of weights
        raise NotImplementedError()

        # offset: int = 0
        # for e in range(self.n_elec + 1):
        #     row = self.W[e, e:e + self.n_orb - self.n_elec + 1]
        #     print(' ' * offset, '---'.join([f'{w:^4}' for w in row]))
        #     offset += 3
        #
        #     if e != self.n_elec:
        #         print(' ' * offset, "\\" + "\\".join([' ' * (4 + 2)] * (self.n_orb - self.n_elec)) + "\\")
        #         offset += 1


SimpleGraphs = dict[tuple[int, int], list[SimpleGraph]]


@attrs.define(eq=True, repr=False)
class DASGraph(Archived):
    """Implements RAS/CAS CI string graph, using SimpleGraph class as a basis.

    TODO: 1c formalism
    TODO: hdf5 storage
    TODO: Excitation lists and string operations
    TODO: extend to more than three ActiveSpaces (generalize spaces and restrictions)
    TODO: implement space division, merging, and other manipulations
    """
    ROOT: ClassVar[str] = "ci/graph"

    spaces: tuple[int, ...] = attrs.field(
        converter=lambda s: tuple(map(int, s)),
        validator=attrs.validators.deep_iterable(
            member_validator=[
                attrs.validators.instance_of(int),
                attrs.validators.ge(0),
            ],
            iterable_validator=attrs.validators.instance_of(tuple),
        ),
        metadata={MCTOOLS_METADATA_KEY: {
            "to_hdf5": np.asarray,
        }}
    )

    reference: tuple[int, ...] = attrs.field(
        converter=lambda s: tuple(map(int, s)),
        validator=attrs.validators.deep_iterable(
            member_validator=[
                attrs.validators.instance_of(int),
                attrs.validators.ge(0),
            ],
            iterable_validator=attrs.validators.instance_of(tuple),
        ),
        metadata={MCTOOLS_METADATA_KEY: {
            "to_hdf5": np.asarray,
        }}
    )

    restrictions: tuple[tuple[int, int], ...] = attrs.field(  # ((1: min_e, max_e), (2: min_e, max_e), ... )
        default=tuple,
        converter=lambda s: tuple(map(lambda t: tuple(map(int, t)), s)),
        validator=attrs.validators.deep_iterable(
            member_validator=[
                attrs.validators.min_len(2),
                attrs.validators.max_len(2),
                attrs.validators.deep_iterable(
                    member_validator=[
                        attrs.validators.instance_of(int),
                        attrs.validators.ge(0),
                    ],
                    iterable_validator=attrs.validators.instance_of(tuple)
                )
            ],
            iterable_validator=attrs.validators.instance_of(tuple),
        ),
        metadata={MCTOOLS_METADATA_KEY: {
            "to_hdf5": np.asarray,
        }}
    )

    max_excitation_level: int = -1
    reverse: bool = False
    cat_order: Literal['F', 'C'] = attrs.field(
        default='F',
        converter=str,
        validator=attrs.validators.in_(['F', 'C'])
    )

    config_dtype: npt.DTypeLike = attrs.field(
        metadata={MCTOOLS_METADATA_KEY: {
            "ignore_hdf5": np.asarray,
        }}
    )

    @config_dtype.default
    def _get_default_config_dtype(self) -> npt.DTypeLike:
        if max(self.spaces) > 64:
            return np.dtype(object)
        return np.dtype(np.int64)

    categories: np.ndarray = attrs.field(default=None, init=False)
    cat_map: dict[tuple[int, int], int] = attrs.field(factory=dict, init=False)
    graphs: SimpleGraphs = attrs.field(factory=dict, init=False)
    n_configs: int = attrs.field(default=0, init=False)

    def __attrs_post_init__(self) -> None:
        if len(self.spaces) != 3:
            raise NotImplementedError()

        if len(self.spaces) != len(self.restrictions) != len(self.reference):
            raise ValueError()

        if any(e > o for e, o in zip(self.reference, self.spaces)):
            raise ValueError('Number of electrons per space cannot exceed number of orbitals in such space')

        if not all(0 <= l <= u <= o for (l, u), o in zip(self.restrictions, self.spaces)):
            raise ValueError()

        self.build_ras_graph()

    @classmethod
    def from_ras_spec(cls, spaces: tuple[int, int, int], n_elec: int, /,
                      max_hole: int = 0, max_elec: int = 0, **kwargs) -> DASGraph:
        reference: tuple[int, int, int] = (spaces[0], n_elec - spaces[0], 0)
        restrictions = (
            (reference[0] - max_hole, reference[0]),
            (0, spaces[1]),
            (0, max_elec)
        )

        return cls(
            spaces=spaces,
            reference=reference,
            restrictions=restrictions,
            **kwargs
        )

    @classmethod
    def from_cas_spec(cls, n_orb: int, n_elec: int, **kwargs) -> DASGraph:
        return cls.from_ras_spec((0, n_orb, 0), n_elec, max_hole=0, max_elec=0, **kwargs)

    # def to_hdf5(self, file: h5py.File, /, prefix: str = '') -> None:
    #     name = '/'.join([prefix, 'graph'])
    #
    #     gr = file.require_group(name)
    #     spaces = np.asarray(self.spaces)
    #     gr.require_dataset(f'spaces', data=spaces, shape=spaces.shape, dtype=spaces.dtype)
    #
    #     reference = np.asarray(self.reference)
    #     gr.require_dataset(f'reference', data=reference, shape=reference.shape, dtype=reference.dtype)
    #
    #     restrictions = np.asarray(self.restrictions)
    #     gr.require_dataset(f'restrictions', data=restrictions, shape=restrictions.shape, dtype=restrictions.dtype)
    #
    #     gr.attrs['max_excitation_level'] = self.max_excitation_level
    #     gr.attrs['reverse'] = self.reverse

    # @classmethod
    # def from_hdf5(cls, file: h5py.File, /, prefix: str = '') -> DASGraph:
    #     name = '/'.join([prefix, 'graph'])
    #     if gr := file.get(name, default=None):
    #         spaces = gr.get('spaces', tuple())
    #         reference = gr.get('reference', tuple())
    #         restrictions = gr.get('restrictions', tuple())
    #         max_excitation_level = gr.attrs.get('max_excitation_level', -1)
    #         reverse = gr.attrs.get('reverse', False)
    #         return cls(
    #             spaces=spaces,
    #             reference=reference,
    #             restrictions=restrictions,
    #             max_excitation_level=max_excitation_level,
    #             reverse=reverse
    #         )
    #     raise KeyError('Graph not Found')

    def build_ras_graph(self) -> NoReturn:
        # Find valid categories
        cat_map: SimpleGraphs = defaultdict(lambda: -1)

        ras1 = self.restrictions[0]
        ras3 = self.restrictions[2]
        for r3_ne, r1_ne in product(range(ras3[-1] + 1), range(ras1[1], ras1[0] - 1, -1)):
            r2_ne = self.n_elec - (r1_ne + r3_ne)
            occ: tuple[int, int, int] = (r1_ne, r2_ne, r3_ne)
            if all(0 <= n_elec <= n_orb
                   for n_orb, n_elec in zip(self.spaces, occ)):
                cat_map[(r1_ne, r3_ne)] = r2_ne

        # FIXME: Use numpy fields for category array, maybe use a pd.DataFrame
        n_cat = len(cat_map)
        offsets = np.zeros(shape=(n_cat, 2 * self.n_spaces + 1), dtype=np.uint64)

        graphs: SimpleGraphs = {}
        graphs_cache: dict[tuple[int, int], SimpleGraph] = {}
        graph_kwargs = dict(reverse=self.reverse, config_dtype=self.config_dtype)

        n_configs = 0
        for cat_idx, ((r1_ne, r3_ne), r2_ne) in enumerate(cat_map.items()):
            space_occ = [r1_ne, r2_ne, r3_ne]

            cat_graphs = [
                graphs_cache.setdefault(
                    (n_orb, n_elec),
                    SimpleGraph(n_orb, n_elec, **graph_kwargs)
                )
                for n_orb, n_elec in zip(self.spaces, space_occ)
            ]

            # Calculate dimensions of each category w.r.t to their RAS
            offsets[cat_idx, :self.n_spaces] = [g.n_configs for g in cat_graphs]

            # Offset of the category
            offsets[cat_idx, self.n_spaces] = n_configs
            n_configs += np.prod(offsets[cat_idx, :self.n_spaces])

            # Restrictions for the category
            offsets[cat_idx, self.n_spaces + 1:] = space_occ

            graphs[(r1_ne, r3_ne)] = cat_graphs
            cat_map[(r1_ne, r3_ne)] = cat_idx
            cat_idx += 1

        self.graphs = graphs
        self.categories = offsets
        self.cat_map = cat_map
        self.n_configs = int(n_configs)

    def get_categories_df(self) -> pd.DataFrame:
        df = pd.DataFrame(self.categories, columns=[
            *[f'n_config_{i + 1}' for i in range(self.n_spaces)],
            'offset',
            *[f'n_elec_{i + 1}' for i in range(self.n_spaces)],
        ])
        return df

    def get_category_configs(self, n_holes: int, n_elec: int) -> ConfigArray:
        cat_spec = self.spaces[0] - n_holes, n_elec
        cat_idx = self.cat_map[cat_spec]

        if cat_idx < 0:
            return np.zeros((0, 0, 0, 0), dtype=self.config_dtype)

        raw_config = []
        for graph in self.graphs[cat_spec]:
            raw_config.append(graph.get_all_configs())

        config = np.array(np.meshgrid(*raw_config, indexing='ij'))
        match self.cat_order:
            case 'C':
                config = config.transpose((1, 2, 3, 0))
            case 'F':
                config = config.transpose((3, 2, 1, 0))

        return config

    def get_category_addrs(self, n_holes: int, n_elec: int) -> AddrArray:
        # TODO: get config addresses for a category
        raise NotImplementedError()

    def get_address(self, config: npt.ArrayLike) -> AddrArray:
        config = np.asarray(config, dtype=self.config_dtype).reshape(-1, self.n_spaces)
        config[..., :] &= self.space_mask

        # Calculate #holes, #electrons (ras2), #electrons (ras3)
        config_pop = get_elec_count(config, config_dtype=self.config_dtype)
        cat_idx = np.apply_along_axis(lambda x: self.cat_map[tuple(x)], 1, config_pop[:, (0, 2)])

        addr = np.zeros_like(config, dtype=np.uint64)
        for cat in np.unique(cat_idx):
            idx = cat_idx == cat
            r1_ne, r2_ne, r3_ne = self.categories[cat, self.n_spaces + 1:2 * self.n_spaces + 1]
            for i, graph in enumerate(self.graphs[(r1_ne, r3_ne)]):
                addr[idx, i] = graph.get_address(config[idx, i])

        # Ravel each address according to its category dimensions
        match self.cat_order:
            case 'F':
                offsets = self.categories[cat_idx, :2]
                addr[:, 1:] *= offsets.cumprod(axis=1)
            case 'C':
                # TODO: Ordering of stings within category in row-major order
                raise NotImplementedError('Raveling in row-major order is nor implemented yet')

        return addr.sum(axis=1) + self.categories[cat_idx, 3].astype(np.uint)

    def get_config(self, addr: npt.ArrayLike) -> ConfigArray:
        raveled_addr = np.asarray(addr, dtype=np.uint64).reshape(-1).copy()

        cat_idx = np.searchsorted(self.categories[:, 3], raveled_addr, side='right') - 1
        raveled_addr -= self.categories[cat_idx, 3]

        # Unravel the address
        addr = np.zeros((len(raveled_addr), 3), dtype=np.uint64)
        match self.cat_order:
            case 'F':
                addr[:, 0] = raveled_addr
                for i in range(1, self.n_spaces):
                    addr[:, i], addr[:, i - 1] = np.divmod(addr[:, i - 1], self.categories[cat_idx, i - 1])

            case 'C':
                # TODO: Ordering of stings within category in row-major order
                raise NotImplementedError('Raveling in row-major order is nor implemented yet')

        config = np.zeros_like(addr, dtype=self.config_dtype)
        for cat in np.unique(cat_idx):
            idx = (cat_idx == cat)
            r1_ne, r2_ne, r3_ne = self.categories[cat, self.n_spaces + 1:2 * self.n_spaces + 1]
            for i, graph in enumerate(self.graphs[(r1_ne, r3_ne)]):
                config[idx, i] = graph.get_config(addr[idx, i])

        return config

    def get_mo_offsets(self) -> np.ndarray:
        return np.cumsum([0, *self.spaces])

    def get_config_repr(self, config: npt.ArrayLike) -> npt.NDArray[np.str_]:
        config = np.asarray(config, dtype=self.config_dtype).reshape(-1, self.n_spaces)

        if self.is_cas:
            return get_config_repr(config[..., 1], (self.n_orb,), config_dtype=self.config_dtype)

        return get_config_repr(config, self.spaces, config_dtype=self.config_dtype)

    @property
    def space_mask(self) -> npt.NDArray[np.uint64]:
        mask = np.asarray(self.spaces, dtype=self.config_dtype)
        if self.config_dtype is object:
            return (1 << mask) - 1

        return (self.config_dtype.type(1) << mask) - self.config_dtype.type(1)

    @property
    def n_orb(self) -> int:
        return sum(self.spaces)

    @property
    def n_elec(self) -> int:
        return sum(self.reference)

    @property
    def n_cat(self) -> int:
        return len(self.categories)

    @property
    def n_spaces(self) -> int:
        return len(self.spaces)

    @property
    def is_cas(self) -> bool:
        return self.n_orb == self.spaces[1]

    @property
    def is_ras(self) -> bool:
        return self.n_spaces == 3

    def get_graph_spec(self) -> str:
        spec: list[str] = []
        for n_orb, (occ_min, occ_max), occ_ref in zip(self.spaces, self.restrictions, self.reference):
            if n_orb < 1:
                continue
            if occ_min == 0 and occ_max == n_orb:
                spec.append(f'{occ_ref:d}e/{n_orb:d}o')
            else:
                max_holes: int = occ_ref - occ_min
                max_particles: int = occ_max - occ_ref

                space_spec: str = ''
                if max_holes:
                    space_spec += f'{max_holes:d}h'

                if max_particles:
                    space_spec += f'{max_particles:d}p'

                space_spec += f'/{n_orb}o'
                spec.append(space_spec)
        return f'{self.n_elec}e{self.n_orb}o|%s' % (', '.join(spec))

    def __repr__(self) -> str:
        ras_spec = self.get_graph_spec()
        return f'{self.__class__.__name__}([{ras_spec}], #Det={self.n_configs:,d}, #Cat={self.n_cat})'


if __name__ == '__main__':
    self = DASGraph.from_ras_spec((12, 50, 10), 12 + 36 + 13, max_hole=1, max_elec=1)
    print(self)
