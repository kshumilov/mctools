import numpy as np
from numpy import typing as npt

from scipy.special import comb

__all__ = [
    'BYTE_TO_BITS',
    'AddrArray',
    'ConfigArray',

    'get_num_configs',
    'get_elec_count',
    'get_configs_reverse',
    'get_configs_direct',

    'get_config_repr'
]


BYTE_TO_BITS = 8

ONE_ULL = np.uint64(1)
ONE_LL = np.int64(1)

AddrArray = npt.NDArray[np.uint64]
ConfigArray = npt.NDArray[np.int64]


def get_num_configs(n_orb: int, n_elec: int, /) -> int:
    return comb(n_orb, min(n_orb, n_elec), exact=True)


def get_configs_reverse(n_orb: int, n_elec: int, /) -> list[int]:
    """Recursively builds the list of configurations for CAS(#e, #o)
    in reversed lexicographical order.

    Each configuration is represented as strings of 0s and 1s,
    with orbitals ordered from right to left.
    Parameters:
        n_orb: number of orbitals in CAS, #o
        n_elec: number of electrons in CAS, #e

    Return:
        List of integers with #e bits set to 1, r

    # TODO: use Numba or Cython to optimize
    """
    if n_elec == 1:
        res = [(1 << o) for o in range(n_orb)]
    elif n_elec >= n_orb:
        n = 0
        for o in range(n_orb):
            n = n | (1 << o)
        res = [n]
    else:
        thisorb = 1 << (n_orb - 1)
        res = get_configs_reverse(n_orb - 1, n_elec)
        for n in get_configs_reverse(n_orb - 1, n_elec - 1):
            res.append(n | thisorb)
    return res


def get_configs_direct(n_orb: int, n_elec: int, /) -> list[int]:
    # TODO: use Numba or Cython to optimize
    if n_elec == 1:
        res = [(1 << o) for o in range(n_orb)]
    elif n_elec >= n_orb:
        n = 0
        for o in range(n_orb):
            n = n | (1 << o)
        res = [n]
    else:
        res = [n << 1 | 1 for n in get_configs_direct(n_orb - 1, n_elec - 1)]
        for n in get_configs_direct(n_orb - 1, n_elec):
            res.append(n << 1)
    return res


def get_elec_count(config: npt.ArrayLike, /, *, config_dtype: npt.DTypeLike = np.int64) -> npt.NDArray[np.uint64]:
    config = np.asarray(config, dtype=config_dtype).copy()

    count = np.zeros_like(config, dtype=np.uint64)
    while (idx := config > 0).any():
        count += idx
        config[idx] &= config[idx] - 1

    return count


def get_config_repr(config: npt.ArrayLike, spaces: tuple[int, ...], /, *,
                    config_dtype: npt.DTypeLike = np.int64) -> npt.NDArray[np.str_]:
    config = np.asarray(config, dtype=config_dtype).reshape(-1, len(spaces))

    def to_str(config_: npt.NDArray[np.unicode]) -> str:
        return ' '.join([np.binary_repr(c, width=n)[::-1]
                         for c, n in zip(config_, spaces)])

    return np.apply_along_axis(to_str, -1, config)
