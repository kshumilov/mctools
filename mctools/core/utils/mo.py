from __future__ import annotations

from functools import reduce
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .constants import ANGULAR_MOMENTUM_SYMBS
from .constants import PeriodicTable as PT

__all__ = [
    'partition_molorb',
    'partorb_to_df',
    'get_natural_orbitals',
]


def _construct_ao_map(df_ao: pd.DataFrame, key: str) -> np.ndarray:
    values = df_ao[key].unique()
    values.sort()

    n_ao = len(df_ao)
    T = np.tile(values[:, np.newaxis], n_ao) == df_ao[key].values
    return T


def partition_molorb(molorb: np.ndarray, df_ao: pd.DataFrame, overlap: np.ndarray, /,
                     by: list[str] | None = None) -> np.ndarray:
    """Partitions MO coefficients according to selected parameter.

    The function works by modifying the orthogonality condition of MO:
        S_MO = C_MO @ S_AO @ C_MO^h, where S_MO is identity, and C_MO^h is adjoint of C_MO

    Args:
        molorb: Numpy array of shape (#MO, #AO) to be partitioned;
        df_ao: Pandas dataframe of length #AO that provides information about AOs;
        overlap: Numpy array of (#AO, #AO) storing AO overlap matrix;

    Keyword Args:
        by: list of columns in df_ao to aggregate by: by=['atom', 'l', ...]

    Returns:
        Numpy array of shape (#MO, #{unique values, in col}) as specified by the inputs.
    """
    n_mo, n_ao = molorb.shape
    if n_ao != len(df_ao):
        raise ValueError("AO mismatch between C and df_ao")

    partitions: list[np.ndarray] = []
    by: list[str] = by if by is not None else []

    for prop_name in by:
        prop = df_ao[prop_name]
        uniques = prop.unique()
        uniques.sort()

        partition: np.ndarray = np.tile(uniques[:, np.newaxis], n_ao) == prop.values
        partitions.append(partition)

    C_inv = overlap @ molorb.T.conj()  # (#AOs, #MOs)

    if partitions:
        partition = reduce(lambda x, y: x[..., np.newaxis, :] * y, partitions)
        partorb = partition @ (molorb.T * C_inv)
        partorb = partorb.transpose(-1, *list(range(len(partitions))))
    else:
        partorb = np.diag(molorb @ C_inv)

    return partorb


def partorb_to_df(partorb: np.ndarray, df_ao: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    by = by if by is not None else []

    if len(by) != partorb.ndim - 1:
        raise ValueError(f'Mismatch between partitioning indices (by) and '
                         f'partitioned orbital shape: {len(by)} != {partorb.ndim - 1}')

    indices = [np.arange(partorb.shape[0])]

    for column in by:
        if column == 'l':
            l = df_ao[column].unique()
            l.sort()
            indices.append(ANGULAR_MOMENTUM_SYMBS[l])
        elif column == 'atom' and 'element' in df_ao:
            atm_el = df_ao[['atom', 'element']].drop_duplicates()
            indices.append([f'{PT.Symbol[t.element - 1]}{t.atom + 1}' for t in atm_el.itertuples()])
        elif column == 'element':
            Z = df_ao[column].unique()
            Z.sort()
            indices.append(PT.Symbol[Z - 1])
        elif column in df_ao:
            values = df_ao[column].unique()
            values.sort()
            indices.append(values)
        else:
            raise ValueError(f'Cannot partition by column: {column}')

    midx = pd.MultiIndex.from_product(indices, names=['mo'] + by)

    df = pd.DataFrame(partorb.flatten(), index=midx, columns=['C'])
    df.reset_index(inplace=True)
    df['C_abs'] = np.abs(df['C'])

    df = df.pivot_table(index='mo', columns=by)
    df['max_C'] = df['C_abs'].max(axis=1)
    df['assign'] = df['C_abs'].idxmax(axis=1)
    # df['element'], df['L'] = zip(*df['assign'])
    return df


def get_natural_orbitals(rdms: np.ndarray, /,
                         state_average: bool = True,
                         weights: Optional[Sequence[float]] = None) -> tuple[np.ndarray, np.ndarray]:
    """Obtains Natural Orbitals by diagonalizing 1RDM.

    One-electron Reduced Density Matrix (1RDM) is defined as (in spinor basis):
        D_pq = <I|q^h p|I>, where I - is the state, and p and q run over MOs.
    The diagonalization of 1RDM in MO bases thus follows:
        U @ s @ U^h = D_pq,
    such that
        - s_p is diagonal matrix of real occupancies: 0 <= s_p <= 1
        - U_pq is special unitary matrix, defining Natural Orbitals of the state I.

    Args:
        rdms: Numpy array of the shape (#MO, #MO) or (#State, #MO, #MO).
        state_average: Whether to perform state average over the provided rdms - default is yes
        weights: Sequence of weights to state average over,
            if state_average == True and weights == None, assumes equal weights for all states
            if state_average == False weights are ignored

    Returns:
        pop: Numpy array of shape (#MO,) or (#State, #MO) with population of Natural MOs
        U: Numpy array of shape (#MO, #MO) or (#State, #MO, #MO), with new natural MOs in MO basis, s.t.
           C_NO = U^h @ C_MO
    """
    match rdms.ndim:
        case 2:
            n_states = 1
        case 3:
            n_states = rdms.shape[0]
        case _:
            raise ValueError('rdms.shape must be equal to 2 or 3')

    if n_states > 1 and state_average:
        weights = np.full(n_states, 1 / n_states) if weights is None else np.asarray(weights)
        if weights.ndim > 1 or len(weights) < n_states:
            raise ValueError('Invalid weights are provided')

        rdms = (weights[:, np.newaxis, np.newaxis] * rdms).sum(axis=0)

    if rdms.shape[-1] != rdms.shape[-2]:
        raise ValueError('rdms must be square hermitian')

    return np.linalg.eigh(rdms)
