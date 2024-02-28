from __future__ import annotations

from enum import StrEnum, IntEnum, unique, Enum, auto
from typing import ClassVar, TypeAlias, Any

import attr
import attrs
import h5py
import numpy as np
import numpy.typing as npt
import pandas as pd

from mctools.core.resource import Resource
from mctools.cli.console import console

from .utils.constants import PeriodicTable

__all__ = [
    'AtomicOrbitalAnsatz',
]


@unique
class AtomicOrbitalAnsatz(StrEnum):
    spherical = auto()
    cartesian = auto()


@unique
class AngularMomentum(IntEnum):
    S = 0
    P = 1
    D = 2
    F = 3
    G = 4
    H = 5
    I = 6

    @classmethod
    def get_label_to_l(cls) -> dict[str, int]:
        return {l.name: l.value for l in list(cls)}

    @classmethod
    def get_l_to_label(cls) -> list[int]:
        return [l.name for l in list(cls)]


def transform_integral(integral: npt.NDArray, transformation: npt.NDArray) -> npt.NDArray:
    return np.einsum('pi,...ij,qj->...pq', transformation, integral, transformation.conj(), optimize='optimal')


Storage: TypeAlias = dict[Resource, Any]


@attrs.define(repr=False, eq=True)
class AtomicOrbitalBasis:
    RESOURCE: ClassVar[Resource] = Resource.ao_basis
    RESOURCE_IDX_COL: ClassVar[str] = 'resource_idx'

    class Col(StrEnum):
        shell_idx = auto()
        atom_idx = auto()
        l = auto()
        ml = auto()
        element = auto()
        x = auto()
        y = auto()
        z = auto()

        @classmethod
        def get_names(cls) -> list[str]:
            return [v.value for v in list(cls)]

        @classmethod
        def get_xyz(cls) -> list[str]:
            return [v.value for v in [cls.x, cls.y, cls.z]]

        @classmethod
        def required(cls) -> list[str]:
            return [v.value for v in [cls.atom_idx, cls.element, cls.l, cls.ml]]

    df: pd.DataFrame = attr.field(validator=attr.validators.instance_of(pd.DataFrame), repr=False)

    ansatz: AtomicOrbitalAnsatz = AtomicOrbitalAnsatz.spherical
    integrals: dict[str, npt.NDArray] = attrs.field(
        factory=dict,
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(str),
            value_validator=attrs.validators.instance_of(np.ndarray),
        )
    )

    def __attrs_post_init__(self):
        match self.ansatz:
            case AtomicOrbitalAnsatz.spherical:
                pass
            case _:
                raise NotImplementedError()

        for integral in self.integrals.values():
            assert integral.ndim == 2
            assert self.n_ao == integral.shape[0]

        for col in self.Col.required():
            assert col in self.df

    @property
    def n_ao(self) -> int:
        return len(self.df)

    @property
    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(#AO={self.n_ao}, ints={self.integrals})'

    @classmethod
    def from_resources(cls, storage: Storage) -> AtomicOrbitalBasis:
        df = cls.build_df(storage)

        # TODO: Validate Integrals
        integrals = {}
        for r in Resource.STV():
            if (integral := storage.get(r, None)) is not None:
                integrals[r.name.split('_')[-1]] = integral

        return cls(df, integrals=integrals)

    @classmethod
    def build_df(cls, storage: Storage) -> pd.DataFrame:
        df_dict = {
            cls.Col.shell_idx.value: storage.get(Resource.ao_basis_shell, None),
            cls.Col.atom_idx.value: storage.get(Resource.ao_basis_atom, None),
            cls.Col.l.value: storage.get(Resource.ao_basis_l, None),
            cls.Col.l.ml.value: storage.get(Resource.ao_basis_ml, None),
        }

        if any(v is None for v in df_dict.values()):
            raise ValueError('Cannot build atomic orbitals')

        if len(set(arr.shape for arr in df_dict.values())) > 1:
            raise ValueError('dimension mismatch')

        df = pd.DataFrame.from_dict(df_dict)

        atomic_numbers = storage[Resource.mol_atnums]
        df[cls.Col.element.value] = PeriodicTable.loc[atomic_numbers[df[cls.Col.atom_idx.value]], 'Symbol'].values

        shell_coords = storage[Resource.ao_basis_shells_coords]
        df[cls.Col.get_xyz()] = shell_coords[df[cls.Col.shell_idx.value]]

        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'ao_idx'
        return df

    def to_hdf5(self, file: h5py.File, /, prefix: str = '') -> None:
        gr = file.require_group('/'.join([prefix, 'ao', 'basis']))
        gr.attrs['ansatz'] = str(self.ansatz.name)

        gr = file.require_group('/'.join([prefix, 'ao', 'basis', 'df']))
        for col_name in self.df.columns:
            col = self.df[col_name]
            values = col.to_numpy(dtype=col.dtype if col.dtype != np.dtype(object) else np.dtype('S'))
            gr.require_dataset(col_name, data=values, dtype=values.dtype, shape=values.shape, compression='gzip')

        gr = file.require_group('/'.join(['ao', 'int1e']))
        for name, array in self.integrals.items():
            gr.require_dataset(name, data=array, shape=array.shape, dtype=array.dtype, compression='gzip')

    @classmethod
    def from_hdf5(cls, file: h5py.File, /, prefix: str = '') -> AtomicOrbitalBasis:
        gr = file.require_group('/'.join([prefix, 'ao', 'basis']))
        ansatz = AtomicOrbitalAnsatz[gr.attrs['ansatz']]

        if (gr := file.get('/'.join([prefix, 'ao', 'basis', 'df']), default=None)) is None:
            raise KeyError()

        # TODO: Validate Columns
        df_dict = {}
        for col_name, col in gr.items():
            df_dict[col_name] = col

        df = pd.DataFrame.from_dict(df_dict)

        # TODO: Validate Integrals
        integrals = {}
        if (gr := file.get('/'.join(['ao', 'int1e']), None)) is None:
            raise KeyError()

        for name, array in gr.items():
            integrals[name] = np.asarray(array)

        return cls(df=df, ansatz=ansatz, integrals=integrals)

    @classmethod
    def get_build_resources(cls) -> Resource:
        return (Resource.ao_basis_shell | Resource.ao_basis_atom | Resource.ao_basis_l | Resource.ao_basis_ml |
                Resource.ao_basis_shells_coords |
                Resource.mol_atnums | Resource.mol_atnums)


@unique
class MolecularOrbitalAnsatz(StrEnum):
    RR = 'Real Restricted'
    CR = 'Complex Restricted'
    RU = 'Real Unrestricted'
    GU = 'Complex Unrestricted'


@attrs.define(repr=True, eq=True)
class MolecularOrbitalBasis:
    RESOURCE: ClassVar[Resource] = Resource.mo_basis
    RESOURCE_IDX_COL: ClassVar[str] = 'resource_idx'

    class Col(StrEnum):
        is_occupied = auto()
        active_space = auto()

        @classmethod
        def get_names(cls) -> list[str]:
            return [v.value for v in list(cls)]

        @classmethod
        def required(cls) -> list[str]:
            return [v.value for v in [cls.is_occupied]]

    df: pd.DataFrame = attr.field(validator=attr.validators.instance_of(pd.DataFrame), repr=False)

    molorb: npt.NDArray = attr.field(validator=attr.validators.instance_of(np.ndarray), repr=False)
    ansatz: MolecularOrbitalAnsatz = MolecularOrbitalAnsatz.GU

    ao_basis: AtomicOrbitalBasis | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.instance_of(AtomicOrbitalBasis))
    )

    def __attrs_post_init__(self) -> None:
        assert self.n_mo == self.molorb.shape[0]

        match self.ansatz:
            case MolecularOrbitalAnsatz.GU:
                assert self.molorb.ndim == 3
                assert self.molorb.shape[-1] == 2
            case _:
                raise NotImplementedError()

        if self.ao_basis is not None:
            assert self.ao_basis.n_ao == self.molorb.shape[1]
            assert 2 == self.molorb.shape[-1]

    @property
    def n_mo(self) -> int:
        return len(self.df)

    @property
    def __len__(self) -> int:
        return len(self.df)

    @classmethod
    def from_resources(cls, storage: Storage) -> MolecularOrbitalBasis:
        df = cls.build_df(storage)

        molorb = storage[Resource.mo_basis_molorb]
        ansatz = storage[Resource.mo_basis_ansatz]
        ao_basis = storage[Resource.ao_basis]

        return cls(df, molorb=molorb, ansatz=ansatz, ao_basis=ao_basis)

    @classmethod
    def build_df(cls, storage: Storage) -> pd.DataFrame:
        n_elec = storage[Resource.mol_nelec]
        n_mo = storage[Resource.mo_basis_molorb].shape[0]
        is_occupied = np.full(n_mo, False, dtype=np.bool_)
        is_occupied[:n_elec] = True

        df_dict = {
            cls.Col.is_occupied.name: is_occupied,
        }

        df = pd.DataFrame.from_dict(df_dict)

        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'mo_idx'
        return df

    def to_hdf5(self, file: h5py.File, /, prefix: str = '') -> None:
        gr = file.require_group('/'.join([prefix, 'mo', 'basis']))
        gr.attrs['ansatz'] = str(self.ansatz.name)

        gr = file.require_group('/'.join([prefix, 'mo', 'basis', 'df']))
        for col_name in self.df.columns:
            col = self.df[col_name]
            gr.require_dataset(col_name, data=col.values, dtype=col.dtype, shape=col.shape, compression='gzip')

        file.require_dataset(
            '/'.join(['mo', 'molorb']),
            data=self.molorb,
            shape=self.molorb.shape,
            dtype=self.molorb.dtype,
            compression='gzip',
        )

        self.ao_basis.to_hdf5(file, prefix=prefix)

    @classmethod
    def get_build_resources(cls) -> Resource:
        return Resource.ao_basis | Resource.mol_nelec

# class Basis:
#     VALID_WF_REFS: ClassVar[list[str]] = ['R', 'U', 'G', 'RO', 'CR']
#     INT_AO_DIMS = np.s_[-2:]
#     MO_AO_DIM = -1
#
#     ELEMENT_COL = 'element'
#
#     __slots__ = [
#         'df', 'molecule',
#         '_integrals',
#
#         'restriction',
#     ]
#
#     df: pd.DataFrame
#     restriction: str
#
#     _integrals: IntegralCollection
#     molecule: Molecule | None
#
#     def __init__(self, df: pd.DataFrame, /,
#                  restriction: str = 'R',
#                  molecule: Molecule | None = None, *,
#                  integrals: IntegralCollection | None = None) -> None:
#         self.df = df
#
#         if restriction not in self.VALID_WF_REFS:
#             raise ValueError(f"Invalid reference restriction: "
#                              f"{self.restriction} not in {self.VALID_WF_REFS}")
#
#         self.restriction = restriction
#         self.molecule = molecule
#
#         integrals = integrals if integrals is not None else {}
#         self._integrals = {}
#         self.add_integrals(integrals)
#
#         if self.molecule is not None:
#             atomic_number = self.molecule.df[self.molecule.ATOMIC_NUMBER_COL].values
#             atom_idx = self.df[self.molecule.ATOM_COL].values
#             self.df[self.ELEMENT_COL] = atomic_number[atom_idx]
#
#     def transform(self, integral: str, trans: npt.NDArray, **kwargs) -> npt.NDArray:
#         aoint = self.get_integral(integral, **kwargs)
#         return transform_integral(aoint, trans)
#
#     def get_aos_ordering(self, ordering: str = 'pyscf') -> npt.NDArray:
#         if ordering != 'pyscf':
#             raise NotImplementedError("Ordering other than 'pyscf' are not implemented")
#
#         aos = self.df.copy(deep=True)
#         new_idx = aos.index.values.copy()
#
#         to_reorder = aos.l > 1
#         old_idx = aos[to_reorder].index.values
#         ordered_aos = aos[to_reorder].sort_values(['atom', 'l', 'shell', 'ml'])
#
#         new_idx[old_idx] = ordered_aos.index.values
#
#         return new_idx
#
#     def add_integrals(self, integrals: IntegralCollection) -> NoReturn:
#         for name, integral in integrals.items():
#             if any(dim != self.n_ao for dim in integral.shape[self.INT_AO_DIMS]):
#                 raise ValueError(f"AO Integral '{name}' has wrong dimensions: "
#                                  f"{integral.shape} != (..., {self.n_ao}, {self.n_ao})")
#             self._integrals[name] = integral
#
#     def get_integral(self, name: 'Basis.Integrals', apply_restriction: bool = True, spin_blocked: bool = False) -> npt.NDArray:
#         if (integral := self._integrals.get(name)) is not None:
#             if apply_restriction:
#                 match self.restriction:
#                     case 'R' | 'RO':
#                         return integral
#                     case 'G':
#                         return self.ao_to_2c(integral, spin_blocked=spin_blocked)
#                     case _:
#                         raise NotImplementedError(f"Transformation to '{self.restriction}' "
#                                                   f"restriction is not implemented.")
#             return integral
#
#         raise KeyError(f"Integral '{name}' is not present in "
#                        f"collection: {self.integrals}")
#
#     def get_aos(self, apply_restriction: bool = True) -> pd.DataFrame:
#         if apply_restriction:
#             match self.restriction:
#                 case 'G':
#                     aos = self.df.loc[self.df.index.repeat(2)].reset_index(drop=True)
#                     aos['spin'] = aos.index % 2
#                     return aos
#                 case _:
#                     raise NotImplementedError(f'Transformation to {self.restriction} '
#                                               f'restriction is not implemented.')
#         return self.df.copy(deep=True)
#
#     def partition_molorb(self, molorb: npt.NDArray, /, overlap_key: Integrals = Integrals.overlap, spin_blocked: bool = False,
#                          to_df: bool = True, **kwargs) -> npt.NDArray | pd.DataFrame:
#         if molorb.shape[self.MO_AO_DIM] != (self.n_ao * self.n_comp):
#             raise ValueError(f'#AOs in C_MO[#MOs, #AOs] does not '
#                              f'match #AOs in the basis: {self.n_ao} != {molorb.shape}')
#
#         S = self.get_integral(overlap_key, spin_blocked=spin_blocked)
#         aos = self.get_aos()
#
#         partorb = partition_molorb(molorb, aos, S, **kwargs)
#
#         if to_df:
#             by = kwargs.get('by', [])
#             return partorb_to_df(partorb, aos, by)
#
#         return partorb
#
#     @classmethod
#     def from_dict(cls, data: ParsingResultType, /,
#                   df_key: str = 'df_ao',
#                   restriction_key: str = 'restriction',
#                   integrals_key: str | list[str] = 'integrals',
#                   skip_missing: bool = False,
#                   instance_key: str = 'basis',
#                   **kwargs):
#         if isinstance(instance := data.get(instance_key, None), cls):
#             return instance
#         elif isinstance(instance, dict):
#             instance = cls.from_dict(instance, **kwargs)
#         elif instance is None:
#             data.update(**kwargs)
#
#             df: pd.DataFrame = data.pop(df_key)
#             restriction = data.pop(restriction_key, None)
#
#             try:
#                 molecule = Molecule.from_dict(data)
#             except (KeyError, ValueError) as exc:
#                 warnings.warn(f'{Molecule.__name__} is not found expect reduced '
#                               f'functionality: {exc.args}', RuntimeWarning)
#                 molecule = None
#
#             integrals: cls.IntegralCollection = {}
#             match integrals_key:
#                 case str(integrals_key):
#                     integrals = data.pop(integrals_key, None)
#                 case [*integral_keys]:
#                     for key in integral_keys:
#                         if (integral := data.pop(key, None)) is not None:
#                             integrals[key] = integral
#                         elif not skip_missing:
#                             raise KeyError(f"Integral '{key}' is not in 'data': {list(data.keys())}")
#
#             instance = data.setdefault(instance_key, cls(df, restriction, molecule, integrals=integrals))
#         else:
#             raise ValueError(f"{cls.__name__} did not recognized '{instance_key}' "
#                              f"item in data: {instance}")
#
#         return instance
#
#     @property
#     def integrals(self) -> set[str]:
#         return set(self._integrals.keys())
#
#     @property
#     def n_ao(self) -> int:
#         return len(self.df)
#
#     @property
#     def n_comp(self) -> int:
#         return 2 if self.restriction == 'G' else 1
#
#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}(#AO={self.n_ao} * {self.n_comp}, ints={self.integrals})'
#
#     @staticmethod
#     def ao_to_2c(integral: npt.NDArray, spin_blocked: bool = True) -> npt.NDArray:
#         I2 = np.eye(2)
#         if spin_blocked:
#             return np.kron(I2, integral)
#         return np.kron(integral, I2)
#
#     @staticmethod
#     def spin_scattered_to_blocked(integral: npt.NDArray) -> npt.NDArray:
#         if integral.ndim < 2:
#             raise ValueError(f"Integral must be at lest 2D: {integral.ndim}")
#
#         if integral.shape[-1] != integral.shape[-2] or integral.shape[-1] % 2:
#             raise ValueError(f"#AO dimensions of must be equal and even: {integral.shape}")
#
#         n_ao = integral.shape[-1] // 2
#         slices = Basis.get_slices(n_ao)
#
#         new_integral = np.zeros_like(integral)
#         for ss, SS in slices:
#             new_integral[SS] = integral[ss]
#
#         return new_integral
#
#     @staticmethod
#     def get_slices(n_ao: int):
#         return (
#             #      s     s              S     S
#             (np.s_[0::2, 0::2], np.s_[:n_ao, :n_ao]),  # aa, AA
#             (np.s_[0::2, 1::2], np.s_[:n_ao, n_ao:]),  # ab, AB
#             (np.s_[1::2, 0::2], np.s_[n_ao:, :n_ao]),  # ba, BA
#             (np.s_[1::2, 1::2], np.s_[n_ao:, n_ao:]),  # bb, BB
#         )
