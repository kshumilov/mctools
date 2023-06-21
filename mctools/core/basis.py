from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar, Type, NoReturn

import numpy as np
import pandas as pd
import numpy.typing as npt

from .molecule import Molecule
from .utils.mo import partition_molorb, partorb_to_df

if TYPE_CHECKING:
    from ..parser.lib import ParsingResult


__all__ = [
    'transform_integral',

    'Basis',
]


def transform_integral(integral: npt.NDArray, transformation: npt.NDArray) -> npt.NDArray:
    return np.einsum('pi,...ij,qj->...pq', transformation, integral, transformation.conj(), optimize='optimal')


class Basis:
    IntegralCollection: ClassVar[Type] = dict[str, npt.NDArray]

    VALID_WF_REFS: ClassVar[list[str]] = ['R', 'U', 'G', 'RO', 'CR']
    INT_AO_DIMS = np.s_[-2:]
    MO_AO_DIM = -1

    ELEMENT_COL = 'element'

    __slots__ = [
        'df', 'molecule',
        '_integrals',

        'restriction',
    ]

    df: pd.DataFrame
    restriction: str

    _integrals: IntegralCollection
    molecule: Molecule | None

    def __init__(self, df: pd.DataFrame, /,
                 restriction: str = 'R',
                 molecule: Molecule | None = None, *,
                 integrals: IntegralCollection | None = None) -> None:
        self.df = df

        if restriction not in self.VALID_WF_REFS:
            raise ValueError(f"Invalid reference restriction: "
                             f"{self.restriction} not in {self.VALID_WF_REFS}")

        self.restriction = restriction
        self.molecule = molecule

        integrals = integrals if integrals is not None else {}
        self._integrals = {}
        self.add_integrals(integrals)

        if self.molecule is not None:
            atomic_number = self.molecule.df[self.molecule.ATOMIC_NUMBER_COL].values
            atom_idx = self.df[self.molecule.ATOM_COL].values
            self.df[self.ELEMENT_COL] = atomic_number[atom_idx]

    def transform(self, integral: str, trans: npt.NDArray, **kwargs) -> npt.NDArray:
        aoint = self.get_integral(integral, **kwargs)
        return transform_integral(aoint, trans)

    def get_aos_ordering(self, ordering: str = 'pyscf') -> npt.NDArray:
        if ordering != 'pyscf':
            raise NotImplementedError("Ordering other than 'pyscf' are not implemented")

        aos = self.df.copy(deep=True)
        new_idx = aos.index.values.copy()

        to_reorder = aos.l > 1
        old_idx = aos[to_reorder].index.values
        ordered_aos = aos[to_reorder].sort_values(['atom', 'l', 'shell', 'ml'])

        new_idx[old_idx] = ordered_aos.index.values

        return new_idx

    def add_integrals(self, integrals: IntegralCollection) -> NoReturn:
        for name, integral in integrals.items():
            if any(dim != self.n_ao for dim in integral.shape[self.INT_AO_DIMS]):
                raise ValueError(f"AO Integral '{name}' has wrong dimensions: "
                                 f"{integral.shape} != (..., {self.n_ao}, {self.n_ao})")
            self._integrals[name] = integral

    def get_integral(self, name: str, apply_restriction: bool = True, spin_blocked: bool = False) -> npt.NDArray:
        if (integral := self._integrals.get(name)) is not None:
            if apply_restriction:
                match self.restriction:
                    case 'R' | 'RO':
                        return integral
                    case 'G':
                        return self.ao_to_2c(integral, spin_blocked=spin_blocked)
                    case _:
                        raise NotImplementedError(f"Transformation to '{self.restriction}' "
                                                  f"restriction is not implemented.")
            return integral

        raise KeyError(f"Integral '{name}' is not present in "
                       f"collection: {self.integrals}")

    def get_aos(self, apply_restriction: bool = True) -> pd.DataFrame:
        if apply_restriction:
            match self.restriction:
                case 'G':
                    aos = self.df.loc[self.df.index.repeat(2)].reset_index(drop=True)
                    aos['spin'] = aos.index % 2
                    return aos
                case _:
                    raise NotImplementedError(f'Transformation to {self.restriction} '
                                              f'restriction is not implemented.')
        return self.df.copy(deep=True)

    def partition_molorb(self, molorb: npt.NDArray, /, overlap_key: str = 'overlap', spin_blocked: bool = False,
                         to_df: bool = True, **kwargs) -> npt.NDArray | pd.DataFrame:
        if molorb.shape[self.MO_AO_DIM] != (self.n_ao * self.n_comp):
            raise ValueError(f'#AOs in C_MO[#MOs, #AOs] does not '
                             f'match #AOs in the basis: {self.n_ao} != {molorb.shape}')

        S = self.get_integral(overlap_key, spin_blocked=spin_blocked)
        aos = self.get_aos()

        partorb = partition_molorb(molorb, aos, S, **kwargs)

        if to_df:
            by = kwargs.get('by', [])
            return partorb_to_df(partorb, aos, by)

        return partorb

    @classmethod
    def from_dict(cls, data: ParsingResult, /,
                  df_key: str = 'df_ao',
                  restriction_key: str = 'restriction',
                  integral_keys: list[str] | None = None,
                  skip_missing: bool = False,
                  instance_key: str = 'basis',
                  **kwargs):
        if isinstance(instance := data.get(instance_key, None), cls):
            return instance
        elif isinstance(instance, dict):
            instance = cls.from_dict(instance, **kwargs)
        elif instance is None:
            data.update(**kwargs)

            df: pd.DataFrame = data.pop(df_key)
            restriction = data.pop(restriction_key, None)

            try:
                molecule = Molecule.from_dict(data)
            except (KeyError, ValueError) as exc:
                warnings.warn(f'{Molecule.__name__} is not found expect reduced '
                              f'functionality: {exc.args}', RuntimeWarning)
                molecule = None

            integrals: cls.IntegralCollection = {}
            integral_keys = integral_keys if integral_keys else []
            for key in integral_keys:
                if (integral := data.pop(key, None)) is not None:
                    integrals[key] = integral
                elif not skip_missing:
                    raise KeyError(f"Integral '{key}' is not in 'data': {list(data.keys())}")

            instance = data.setdefault(instance_key, cls(df, restriction, molecule, integrals=integrals))
        else:
            raise ValueError(f"{cls.__name__} did not recognized '{instance_key}' "
                             f"item in data: {instance}")

        return instance

    @property
    def integrals(self) -> set[str]:
        return set(self._integrals.keys())

    @property
    def n_ao(self) -> int:
        return len(self.df)

    @property
    def n_comp(self) -> int:
        return 2 if self.restriction == 'G' else 1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(#AO={self.n_ao} * {self.n_comp}, ints={self.integrals})'

    @staticmethod
    def ao_to_2c(integral: npt.NDArray, spin_blocked: bool = True) -> npt.NDArray:
        I2 = np.eye(2)
        if spin_blocked:
            return np.kron(I2, integral)
        return np.kron(integral, I2)

    @staticmethod
    def spin_scattered_to_blocked(integral: npt.NDArray) -> npt.NDArray:
        if integral.ndim < 2:
            raise ValueError(f"Integral must be at lest 2D: {integral.ndim}")

        if integral.shape[-1] != integral.shape[-2] or integral.shape[-1] % 2:
            raise ValueError(f"#AO dimensions of must be equal and even: {integral.shape}")

        n_ao = integral.shape[-1] // 2
        slices = Basis.get_slices(n_ao)

        new_integral = np.zeros_like(integral)
        for ss, SS in slices:
            new_integral[SS] = integral[ss]

        return new_integral

    @staticmethod
    def get_slices(n_ao: int) -> tuple[tuple[slice, slice], ...]:
        return (
            #      s     s              S     S
            (np.s_[0::2, 0::2], np.s_[:n_ao, :n_ao]),  # aa, AA
            (np.s_[0::2, 1::2], np.s_[:n_ao, n_ao:]),  # ab, AB
            (np.s_[1::2, 0::2], np.s_[n_ao:, :n_ao]),  # ba, BA
            (np.s_[1::2, 1::2], np.s_[n_ao:, n_ao:]),  # bb, BB
        )
