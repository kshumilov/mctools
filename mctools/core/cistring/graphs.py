import warnings
from collections import defaultdict
from itertools import product
from typing import NamedTuple, NoReturn, Literal

import numpy as np
import numpy.typing as npt

from scipy.special import comb

from .utils import *

__all__ = [
    'RASMOs', 'Electrons',
    'SimpleGraph', 'RASGraph',
]


class SimpleGraph:
    """Basic CI string manipulations such as addressing, excitation lists, etc.

    In this implementations, the edge unoccupied edges of the graph are taken to be zero.
    This means that only occupied orbitals play roles in formation of the CI configuration address.

    Attributes:
        n_orb: int --- # Number of orbitals in the space
        n_elec: int --- # Number of electrons in the space

    TODO: Rewrite nodes and edges formation in rectangle format (i.e. transform parallelogram graph into rectangle)
    TODO: implement <I|J>, p^+|J>, p|J>, excitation lists, etc...
    TODO: Fix dtypes and typing
    TODO: allow python int type for configurations
    TODO: Use dtype with multiple int fields to manipulate strings
    """
    __slots__ = [
        'n_orb', 'n_elec',
        'n_configs', 'reverse',
        'edges', 'offsets',

        'config_dtype',
        'max_orb'
    ]

    n_orb: int
    n_elec: int

    n_configs: int  # Number of determinants = n_orb choose n_e
    reverse: bool

    nodes: npt.NDArray[np.uint64]
    edges: npt.NDArray[np.uint64]
    offsets: npt.NDArray[np.uint64]

    dtype: npt.DTypeLike = np.dtype(np.uint64)
    # config_dtype: ClassVar[npt.DTypeLike] = np.dtype(np.int64)
    # max_orb: ClassVar[np.int64] = dtype.type(config_dtype.itemsize * BYTE_TO_BITS)

    def __init__(self, n_orb: int, n_elec: int, /, nodes: npt.NDArray[np.uint64] | None = None, *,
                 reverse: bool = False, use_python_int: bool = False):
        if use_python_int:
            warnings.warn('Using python int as config data type, expect slow performance', RuntimeWarning)
            self.config_dtype = np.dtype(object)
            self.max_orb = np.inf
        else:
            self.config_dtype = np.dtype(np.int64)
            self.max_orb = self.config_dtype.itemsize * BYTE_TO_BITS

        if n_orb > self.max_orb:
            raise ValueError(f'{self.__class__.__name__} is unable to support space with {n_orb} orbitals. '
                             f'Current maximum is {self.max_orb} orbitals.')

        if not (n_orb >= n_elec >= 0):
            raise ValueError('Invalid definition of active space')

        self.n_orb = n_orb
        self.n_elec = n_elec
        self.reverse = reverse

        self.n_configs = get_num_configs(self.n_orb, self.n_elec)
        if self.n_configs == 0:
            spec = self.get_graph_spec()
            warnings.warn(f'Graph {spec} is emtpy', RuntimeWarning)

        nodes = self.get_nodes() if nodes is None else nodes
        if self.reverse:
            if self.n_configs != nodes[-1, -1]:
                raise ValueError(f'Provided nodes might be indexing strings in direct order.')
        else:
            if self.n_configs != nodes[0, 0]:
                raise ValueError(f'Provided nodes might be indexing strings in reverse order.')

        self.edges = self.get_edges(nodes)
        self.offsets = self.get_offsets(self.edges)

    def get_edges(self, nodes: npt.NDArray[np.uint64], /) -> npt.NDArray[np.uint64]:
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
        """Builds weights of nodes in graph according to the given ordering.

        In 'reverse' ordering paths run (0, 0) -> (e, o), while in 'direct' (#e, #o) -> (e, o).
        The weight of a node is thus can be written as:
            nodes[e, o] = (len(path) choose #diagonals steps)
                        = (o choose e) <--> reverse
                        = (#o - o choose #e - e) <--> direct
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

        For example, here is directly ordered weights of CAS(7o, 3e) in Kramer's unrestricted formalism:
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

        addr = np.zeros_like(config, dtype=self.dtype)

        o = 0
        e = np.zeros_like(config)
        bit = np.zeros_like(addr, dtype=self.config_dtype)

        while (idx := e < self.n_elec).any():
            # bit = ((config[idx] >> o) & ONE).astype(self.dtype)
            np.right_shift(config, o, out=bit, where=idx, dtype=self.config_dtype, casting='unsafe')
            np.bitwise_and(bit, 1, out=bit, where=idx, dtype=self.config_dtype)

            # e[idx] += bit
            np.add(e, bit, out=e, where=idx, dtype=e.dtype, casting='unsafe')

            addr[idx] += (self.edges[e[idx] - 1, o] * bit[idx]).astype(addr.dtype)

            o += 1
        return addr

    def get_config(self, addr: np.ndarray) -> np.ndarray:
        addr = np.asarray(addr, dtype=self.dtype).copy()
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

        # if self.config_dtype is object:
        #     return int((1 << self.n_orb) - 1)
        # return (self.config_dtype.type(1) << self.config_dtype.type(self.n_orb)) - self.config_dtype.type(1)

    def get_graph_spec(self) -> str:
        return f'[{self.n_elec:>2d}e, {self.n_orb:>2d}o]'

    def __repr__(self) -> str:
        spec = self.get_graph_spec()
        return f'{self.__class__.__name__}({spec}, #Det={self.n_configs:,d})'

    def __eq__(self, other: 'SimpleGraph') -> bool:
        return self.n_orb == other.n_orb and \
               self.n_elec == other.n_elec and \
               self.reverse == other.reverse and \
               self.config_dtype == other.config_dtype

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


SimpleGraphs = dict[tuple[int, int], list[SimpleGraph]]


class RASGraph:
    """Implements RAS/CAS CI string graph, using SimpleGraph class as a basis.

    TODO: 1c formalism
    TODO: hdf5 storage
    TODO: Excitation lists and string operations
    TODO: extend to more than three ActiveSpaces (generalize spaces and restrictions)
    TODO: implement space division, merging, and other manipulations
    """
    __slots__ = [
        'spaces', 'elec',
        'max_hole', 'max_elec', 'n_configs',
        'reverse', 'cat_order',
        'graphs', 'categories', 'cat_map',
        'config_dtype'
    ]

    spaces: RASMOs
    elec: Electrons

    max_hole: int
    max_elec: int

    reverse: bool
    cat_order: Literal['F', 'C']

    n_configs: int

    graphs: SimpleGraphs
    categories: np.ndarray
    cat_map: dict[tuple[int, int], int]
    config_dtype: npt.DTypeLike

    def __init__(self, spaces: RASMOs, elec: Electrons, max_hole: int, max_elec: int, /,
                 reverse: bool = False, cat_order: Literal['F', 'C'] = 'F', use_python_int: bool = False):
        assert all(r >= 0 for r in spaces)
        assert all(e >= 0 for e in elec)

        self.spaces = spaces
        self.elec = elec

        assert max_hole >= 0
        assert max_elec >= 0

        assert max_elec <= self.n_elec
        assert spaces.r1 >= max_hole
        assert spaces.r1 <= self.n_elec
        assert spaces.r3 >= max_elec

        self.max_hole = max_hole
        self.max_elec = max_elec

        assert cat_order in 'FC'
        self.cat_order = cat_order
        self.reverse = reverse

        # TODO: use numpy structured dtype in the future
        if use_python_int:
            warnings.warn('Using python int as config data type, expect slow performance', RuntimeWarning)
            self.config_dtype = np.dtype(object)
        else:
            self.config_dtype = np.dtype(np.int64)

        self._build_graph(reverse=reverse, use_python_int=use_python_int)

    def _build_graph(self, reverse: bool = False, use_python_int: bool = False) -> NoReturn:
        # Find valid categories
        cat_map = defaultdict(lambda: -1)
        for r3_ne, r1_nh in product(range(self.max_elec + 1), range(self.max_hole + 1)):
            r1_ne = self.spaces.r1 - r1_nh
            if (r2_ne := self.n_elec - (r1_ne + r3_ne)) >= 0:
                cat_map[(r1_ne, r3_ne)] = r2_ne

        # FIXME: Use numpy fields for category array, maybe use a pd.DataFrame
        n_cat = len(cat_map)
        offsets = np.zeros(shape=(n_cat, 2 * self.n_spaces + 1), dtype=SimpleGraph.dtype)

        graphs: SimpleGraphs = {}
        graphs_cache: dict[tuple[int, int], SimpleGraph] = {}
        graph_kwargs = dict(reverse=reverse, use_python_int=use_python_int)

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

    def get_category_configs(self, n_holes: int, n_elec: int) -> ConfigArray:
        cat_spec = self.spaces.r1 - n_holes, n_elec
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

        addr = np.zeros_like(config, dtype=SimpleGraph.dtype)
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
        raveled_addr = np.asarray(addr, dtype=SimpleGraph.dtype).reshape(-1).copy()

        cat_idx = np.searchsorted(self.categories[:, 3], raveled_addr, side='right') - 1
        raveled_addr -= self.categories[cat_idx, 3]

        # Unravel the address
        addr = np.zeros((len(raveled_addr), 3), dtype=SimpleGraph.dtype)
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
            return get_config_repr(config[..., 1], (self.n_mo,), config_dtype=self.config_dtype)

        return get_config_repr(config, self.spaces, config_dtype=self.config_dtype)

    @property
    def space_mask(self) -> npt.NDArray[SimpleGraph.dtype]:
        mask = np.asarray(self.spaces, dtype=self.config_dtype)
        if self.config_dtype is object:
            return (1 << mask) - 1

        return (self.config_dtype.type(1) << mask) - self.config_dtype.type(1)

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

    @property
    def is_1c(self) -> bool:
        return self.n_elec != self.elec.alpha or \
               self.n_elec != self.elec.beta

    def get_graph_spec(self) -> str:
        if self.is_cas:
            spec = f'{self.n_mo}o'
        else:
            restrictions = [-self.max_hole, 0, self.max_elec]
            spec = ','. join([f'{r:>-2d}/{mo:>2d}' for r, mo in zip(restrictions, self.spaces)])
        return f'{self.n_elec}e|{spec}'

    def __repr__(self) -> str:
        ras_spec = self.get_graph_spec()
        return f'{self.__class__.__name__}([{ras_spec}], #Det={self.n_configs:,d}, #Cat={self.n_cat})'

    def __eq__(self, other: 'RASGraph') -> bool:
        return self.spaces == other.spaces and \
               self.elec == other.elec and \
               self.max_elec == other.max_elec and \
               self.max_elec == other.max_elec and \
               self.reverse == other.reverse and \
               self.cat_order == self.cat_order


if __name__ == '__main__':
    ras = RASMOs(36, 14, 12)
    elec = Electrons(36, 0)
    graph = RASGraph(ras, elec, 2, 2)

    configs = graph.get_config([204391, 203794])
    print(graph.get_config_repr(configs))
    print()

    addrs = graph.get_address(configs)
    new_configs = graph.get_config(addrs)
    print(graph.get_config_repr(new_configs))

    ras = RASMOs(0, 68, 0)
    elec = Electrons(67, 0)
    graph = RASGraph(ras, elec, 0, 0, use_python_int=True)

    configs = graph.get_category_configs(0, 0)
    print(graph.get_config_repr(configs))
    print()

    addrs = graph.get_address(configs)
    new_configs = graph.get_config(addrs)
    print(graph.get_config_repr(new_configs))
    # g = CIGraph(7, 3)
    # W = g.get_nodes()
    # Y = g.get_edges(W)
    #
    # c = g.get_all_configs()
    # a = g.get_address(c)
    # r = g.get_config_repr(c)
    # c_new = g.get_config(a)
    #
    # Yp = np.zeros_like(Y)
    # Yp[-1] = Y[-1]
    # print(Yp)
    # print()
    # for i in range(-1, -Y.shape[0] - 1, -1):
    #     print(i)
    #     print(f'ith + 1 = {i + 1}')
    #     print(Yp[i + 1])
    #     print('rolled')
    #     print(np.roll(Yp[i + 1], -1))
    #     print('+')
    #     print(Y[i])
    #     Yp[i] = np.roll(Yp[i + 1], -1) + Y[i]
    #     print('final')
    #     print(Yp)
    #     print()
