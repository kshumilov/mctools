from itertools import product
from dataclasses import dataclass, field, InitVar
from typing import NamedTuple, NoReturn, Tuple, Optional, Literal, ClassVar, Any

import numpy as np
import numpy.typing as npt

from scipy.special import comb

from .utils import *

__all__ = [
    'RASMOs', 'Electrons',
    'CIGraph', 'RASGraph',
]


@dataclass(slots=True)
class CIGraph:
    """Implements basic CI string manipulations such as addressing, excitation lists, etc.

    TODO: rewrite conversion using hdf5 format
    TODO: implement <I|J>, p^+|J>, p|J>, excitation lists, etc...
    TODO: (maybe) Implement as a subclass of np.ndarray
    TODO: Fix dtypes and typing
    TODO: allow python int type for configurations
    TODO: Use dtype with multiple int fields to manipulate strings
    """
    n_orb: int
    n_elec: int

    reverse: bool = False

    nodes: InitVar[Optional[npt.NDArray[np.uint64]]] = field(default=None)
    Y: npt.NDArray[np.uint64] = field(init=False, repr=False)

    n_configs: int = field(init=False)

    # TODO: Use occupation lists
    config_dtype: ClassVar[npt.DTypeLike] = np.dtype(np.int64)
    dtype: ClassVar[npt.DTypeLike] = np.dtype(np.uint64)
    max_orb: ClassVar[np.int64] = dtype.type(config_dtype.itemsize * BYTE_TO_BITS)

    def __post_init__(self, nodes: npt.NDArray[np.uint64] | None):
        assert self.n_orb <= self.max_orb

        if not self.n_orb >= self.n_elec >= 0:
            raise ValueError('Invalid definition of active space')

        nodes = self.get_nodes() if nodes is None else nodes
        self.n_configs = comb(self.n_orb, self.n_elec, exact=True)

        if self.reverse:
            assert self.n_configs == nodes[-1, -1]
        else:
            assert self.n_configs == nodes[0, 0]

        self.Y = self.get_edges(nodes)

    def get_edges(self, nodes: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint]:
        Y = np.zeros((self.n_elec, self.n_orb), dtype=self.dtype)

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
        """Builds weights of nodes in graph according to the ordering.

        In 'reverse' ordering paths run (0, 0) -> (e, o), while in 'direct' (#e, #o) -> (e, o).
        The weight of a node is thus can be written as:
            nodes[e, o] = (len(path) choose #diagonals steps)
                        = (o choose e) --- reverse
                        = (#o - o choose #e - e) --- direct
        """
        nodes = np.zeros((self.n_elec + 1, self.n_orb + 1), dtype=self.dtype)

        idx = self.get_node_idx()
        if self.reverse:
            for e, o in zip(*idx):
                nodes[e, o] = comb(o, e, exact=True)
        else:
            for e, o in zip(*idx):
                nodes[e, o] = comb(self.n_orb - o, self.n_elec - e, exact=True)

        return nodes

    def get_node_idx(self) -> tuple[np.ndarray, np.ndarray]:
        """Builds indices of non-zero elements of the graph.

        For convenience the Full CI graph is stored in (#elec + 1, #orb + 1)-shaped arrays.
        However, there are only (#elec + 1, #orb - #elec + 1) non-zero elements
        which form a slanted array.

        For example, here is directly ordered weights of CAS(7, 3):
              <--#o - #e + 1-->
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
        e_idx, o_idx = np.indices((self.n_elec + 1, l_o + 1), dtype=np.uint8)
        o_idx += e_idx
        return e_idx.ravel(), o_idx.ravel()

    def get_address(self, config: ConfigArray) -> AddrArray:
        config = np.asarray(config, dtype=self.dtype)
        config &= self.space_mask

        addr = np.zeros_like(config, dtype=self.dtype)

        o = self.dtype.type(0)
        e = np.zeros_like(config, dtype=self.dtype)
        while (idx := e < self.n_elec).any():
            bit = (config[idx] >> o) & self.dtype.type(1)
            e[idx] += bit
            addr[idx] += self.Y[e[idx] - 1, o] * bit
            o += self.dtype.type(1)
        return addr

    def get_config(self, addr: np.ndarray[Any, dtype]) -> np.ndarray[Any, config_dtype]:
        addr = np.asarray(addr, dtype=self.dtype).copy()
        config = np.zeros_like(addr, dtype=self.config_dtype)

        e_idx, o_idx = self.get_node_idx()
        e = self.n_elec
        o_curr = np.full_like(addr, self.n_orb, dtype=self.config_dtype)
        while e > 0:
            o = o_idx[e_idx == e] - 1
            for i in range(len(o_curr)):
                j = np.searchsorted(addr[i] < self.Y[e - 1, e - 1:o_curr[i]], False, side='right')
                o_curr[i] = o[j - 1]

            addr -= self.Y[e - 1, o_curr]
            config |= 1 << o_curr
            e -= 1
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
        return get_config_repr(config, (self.n_orb,))

    @property
    def space_mask(self) -> int:
        return (self.config_dtype.type(1) << self.config_dtype.type(self.n_orb)) - self.config_dtype.type(1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.n_elec:>2d}e, {self.n_orb:>2d}o, #Det={self.n_configs})'

    def __eq__(self, other: 'CIGraph') -> bool:
        return self.n_orb == other.n_orb and \
               self.n_elec == other.n_elec and \
               self.reverse == other.reverse

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


class RASMOs(NamedTuple):
    r1: int  # #MOs in RAS1
    r2: int  # #MOs in RAS2
    r3: int  # #MOs in RAS3


class Electrons(NamedTuple):
    alpha: int  # #alpha electrons
    beta: int  # #beta electrons


@dataclass(slots=True)
class RASGraph:
    """Implements RAS/CAS CI string graph, using CIGraph class as a basis.

    TODO: 1c formalism
    TODO: hdf5 storage
    TODO: Excitation lists and string operations
    TODO: extend to more than three ActiveSpaces (generalize spaces and restrictions)
    TODO: implement space division, merging, and other manipulations
    """
    spaces: RASMOs
    elec: Electrons

    max_hole: int
    max_elec: int

    reverse: bool = False
    cat_order: Literal['F', 'C'] = 'F'

    n_configs: int = field(init=False)

    graphs: list[list[CIGraph]] = field(init=False, repr=False)
    categories: np.ndarray = field(init=False, repr=False)
    restriction_map: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> NoReturn:
        assert all(r >= 0 for r in self.spaces)
        assert all(e >= 0 for e in self.elec)
        assert self.max_hole >= 0
        assert self.max_elec >= 0

        assert self.max_elec <= self.n_elec
        assert self.spaces.r1 >= self.max_hole
        assert self.spaces.r1 <= self.n_elec
        assert self.spaces.r3 >= self.max_elec

        assert self.cat_order in 'FC'

        self.graphs = self.get_ras_graphs()
        self.categories, self.restriction_map, self.n_configs = self.get_categories()

    def get_ras_graphs(self) -> list[list[CIGraph]]:
        graphs: list[list[CIGraph]] = [[], [], []]

        # RAS1
        for n_h in range(self.max_hole + 1):
            graphs[0].append(CIGraph(self.spaces.r1, self.spaces.r1 - n_h, reverse=self.reverse))

        # RAS2
        for n_e in range(self.spaces.r2 + 1):
            graphs[1].append(CIGraph(self.spaces.r2, n_e, reverse=self.reverse))

        # RAS3
        for n_p in range(self.max_elec + 1):
            graphs[2].append(CIGraph(self.spaces.r3, n_p, reverse=self.reverse))

        return graphs

    def get_categories(self) -> Tuple[npt.NDArray[CIGraph.dtype], npt.NDArray[CIGraph.dtype], int]:
        holes, elecs = np.arange(self.max_hole + 1), np.arange(self.max_elec + 1)
        restriction_map = self.n_elec - (self.spaces.r1 - holes) - elecs[:, np.newaxis]
        n_cat = np.count_nonzero((self.spaces.r2 >= restriction_map) & (restriction_map >= 0))

        # FIXME: Use numpy fields for category array, maybe use a pd.DataFrame
        offsets = np.zeros(shape=(n_cat, 7), dtype=CIGraph.dtype)

        cat_idx, n_configs = 0, 0
        for ie, ih in product(elecs, holes):
            n_e = restriction_map[ie, ih]
            if 0 <= n_e <= self.spaces.r2:
                # Calculate dimensions of each category w.r.t to their RAS
                offsets[cat_idx, 0] = self.graphs[0][ih].n_configs
                offsets[cat_idx, 1] = self.graphs[1][n_e].n_configs
                offsets[cat_idx, 2] = self.graphs[2][ie].n_configs

                # Offset of the category
                offsets[cat_idx, 3] = n_configs
                n_configs += np.prod(offsets[cat_idx, :3])

                # Restrictions for the category
                offsets[cat_idx, 4] = ih
                offsets[cat_idx, 5] = n_e
                offsets[cat_idx, 6] = ie

                restriction_map[ie, ih] = cat_idx
                cat_idx += 1

        return offsets, restriction_map, int(n_configs)

    def get_category_configs(self, n_holes: int, n_elec: int) -> ConfigArray:
        cat_idx = self.restriction_map[n_elec, n_holes]

        if cat_idx < 0:
            return np.zeros((0, 0, 0, 0), dtype=CIGraph.config_dtype)

        ih, n_e, ie = self.categories[cat_idx, 4:7]

        r1_graph = self.graphs[0][ih]
        r1_configs = r1_graph.get_all_configs()

        r2_graph = self.graphs[1][n_e]
        r2_configs = r2_graph.get_all_configs()

        r3_graph = self.graphs[2][ie]
        r3_configs = r3_graph.get_all_configs()

        configs = np.array(np.meshgrid(r1_configs, r2_configs, r3_configs, indexing='ij'))

        match self.cat_order:
            case 'C':
                configs = configs.transpose((1, 2, 3, 0))
            case 'F':
                configs = configs.transpose((3, 2, 1, 0))
        return configs

    def get_category_addrs(self, n_holes: int, n_elec: int) -> AddrArray:
        # TODO: get config addresses for a category
        raise NotImplementedError()
        # cat_idx = self.restriction_map[n_elec, n_holes]
        # if cat_idx < 0:
        #     return np.zeros((0,), dtype=CIGraph.config_dtype)

    def get_address(self, config: npt.ArrayLike) -> AddrArray:
        config = np.asarray(config, dtype=CIGraph.config_dtype).reshape(-1, self.n_spaces)
        config[..., :] &= self.space_mask

        # Calculate #holes, #electrons (ras2), #electrons (ras3)
        config_pop = get_elec_count(config)
        config_pop[:, 0] = self.spaces.r1 - config_pop[:, 0]
        cat_idx = self.restriction_map[config_pop[:, 2], config_pop[:, 0]]

        addr = np.zeros_like(config, dtype=CIGraph.dtype)
        for cat in np.unique(cat_idx):
            idx = cat_idx == cat

            ih, n_e, ie = self.categories[cat, 4:7]
            addr[idx, 0] = self.graphs[0][ih].get_address(config[idx, 0])
            addr[idx, 1] = self.graphs[1][n_e].get_address(config[idx, 1])
            addr[idx, 2] = self.graphs[2][ie].get_address(config[idx, 2])

        # Ravel each address according to its category dimensions
        match self.cat_order:
            case 'F':
                offsets = self.categories[cat_idx, :2]
                addr[:, 1:] *= offsets.cumprod(axis=1)
            case 'C':
                # FIXME: Ordering of stings within category in row-major order
                raise NotImplementedError('Raveling in row-major order is nor implemented yet')

        return addr.sum(axis=1) + self.categories[cat_idx, 3].astype(np.uint)

    def get_config(self, addr: npt.ArrayLike) -> ConfigArray:
        raveled_addr = np.asarray(addr, dtype=CIGraph.dtype).reshape(-1).copy()

        cat_idx = np.searchsorted(self.categories[:, 3], raveled_addr, side='right') - 1
        raveled_addr -= self.categories[cat_idx, 3]

        # Unravel the address
        addr = np.zeros((len(raveled_addr), 3), dtype=CIGraph.dtype)
        match self.cat_order:
            case 'F':
                addr[:, 0] = raveled_addr

                addr[:, 1] = raveled_addr // self.categories[cat_idx, 0]
                addr[:, 0] = raveled_addr % self.categories[cat_idx, 0]

                addr[:, 2] = addr[:, 1] // self.categories[cat_idx, 1]
                addr[:, 1] = addr[:, 1] % self.categories[cat_idx, 1]
            case 'C':
                # FIXME: Ordering of stings within category in row-major order
                raise NotImplementedError('Raveling in row-major order is nor implemented yet')

        config = np.zeros_like(addr, dtype=CIGraph.config_dtype)
        for cat in np.unique(cat_idx):
            idx = cat_idx == cat

            ih, n_e, ie = self.categories[cat, 4:7]
            config[idx, 0] = self.graphs[0][ih].get_config(addr[idx, 0])
            config[idx, 1] = self.graphs[1][n_e].get_config(addr[idx, 1])
            config[idx, 2] = self.graphs[2][ie].get_config(addr[idx, 2])

        return config

    def get_mo_offsets(self) -> np.ndarray:
        return np.cumsum([0, *self.spaces])

    def get_config_repr(self, config: npt.ArrayLike) -> npt.NDArray[np.str_]:
        config = np.asarray(config, dtype=CIGraph.config_dtype).reshape(-1, self.n_spaces)

        if self.is_cas:
            return get_config_repr(config[..., 1], (self.n_mo,))

        return get_config_repr(config, self.spaces)

    @property
    def space_mask(self) -> npt.NDArray[CIGraph.dtype]:
        mask = np.asarray(self.spaces, dtype=CIGraph.config_dtype)
        mask = (CIGraph.config_dtype.type(1) << mask) - CIGraph.dtype.type(1)
        return mask

    @property
    def n_elec(self) -> int:
        return sum(self.elec)

    @property
    def n_cat(self) -> int:
        return len(self.categories)

    @property
    def n_mo(self) -> int:
        return sum(self.spaces)

    @property
    def n_spaces(self) -> int:
        return len(self.spaces)

    @property
    def is_cas(self) -> bool:
        return self.n_mo == self.spaces[1]

    def get_graph_spec(self) -> str:
        if self.is_cas:
            spec = f'{self.n_mo}o'
        else:
            restrictions = [-self.max_hole, 0, self.max_elec]
            spec = ','. join([f'{r:>-2d}/{mo:>2d}' for r, mo in zip(restrictions, self.spaces)])
        return f'{self.n_elec}e|{spec}'

    def __repr__(self) -> str:
        ras_spec = self.get_graph_spec()
        return f'{self.__class__.__name__}([{ras_spec}], #Det={self.n_configs:_d}, #Cat={self.n_cat})'

    def __eq__(self, other: 'RASGraph') -> bool:
        return self.spaces == other.spaces and \
               self.elec == other.elec and \
               self.max_elec == other.max_elec and \
               self.max_elec == other.max_elec and \
               self.reverse == other.reverse and \
               self.cat_order == self.cat_order


if __name__ == '__main__':
    ras = RASMOs(12, 14, 10)
    elec = Electrons(13, 0)
    graph = RASGraph(ras, elec, 2, 2)
    cnfgs = np.asarray([[4094, 8192, 256]], dtype=CIGraph.config_dtype)
    addr = graph.get_address(cnfgs)
    print(addr)

    new_configs = graph.get_config(addr)
    print(new_configs)
