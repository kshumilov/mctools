from __future__ import annotations

import pathlib
from enum import unique, StrEnum, auto
from typing import TypeAlias, Any, ClassVar

import attrs
import h5py
import numpy as np
import pandas as pd
from numpy import typing as npt

from mctools.core.utils.constants import PeriodicTable, ANGULAR_MOMENTUM_SYMBS, I2
from mctools.newcore.consolidator import Consolidator
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY
from mctools.newcore.resource import Resource

__all__ = [
    'AtomicOrbitalAnsatz',
    'AtomicOrbitalBasis',
    'MolecularOrbitalAnsatz',
    'MolecularOrbitalBasis',
]


@unique
class AtomicOrbitalAnsatz(StrEnum):
    spherical = auto()
    cartesian = auto()


def transform_integral(integral: npt.NDArray, transformation: npt.NDArray) -> npt.NDArray:
    return np.einsum('pi,...ij,qj->...pq', transformation, integral, transformation.conj(), optimize='optimal')


Resources: TypeAlias = dict[Resource, Any]


@attrs.define(repr=False, eq=True)
class AtomicOrbitalBasis(Consolidator):
    RESOURCE: ClassVar[Resource] = Resource.ao_basis
    ROOT: ClassVar[pathlib.Path] = '/'.join(['ao', 'basis'])

    TABLE_SCHEMA: ClassVar[dict[str | list[str, ...], dict]] = {
        'shell_idx': {
            'resource': Resource.ao_basis_shell,
        },
        'atom_idx': {
            'resource': Resource.ao_basis_atom,
        },
        'l': {
            'resource': Resource.ao_basis_l,
        },
        'ml': {
            'resource': Resource.ao_basis_ml,
        },
        ('x', 'y', 'z'): {
            'resource': Resource.ao_basis.ao_basis_shells_coords,
        },
        'element': {
            'resource': Resource.mol_atnums,
            'required': False,
            'factory': lambda df, r: PeriodicTable.loc[r[df['atom_idx']], 'Symbol'].values
        },
        'L': {
            'required': False,
            'factory': lambda df: ANGULAR_MOMENTUM_SYMBS[df['l']]
        },
        'atom': {
            'required': False,
            'factory': lambda df: df['element'] + df['atom_idx'].astype(np.str_)
        }
    }

    class Col(StrEnum):
        shell_idx = auto()
        atom_idx = auto()
        atom = auto()
        l = 'l'
        L = 'L'
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

    ansatz: AtomicOrbitalAnsatz = attrs.field(
        default=AtomicOrbitalAnsatz.spherical,
        converter=lambda v: AtomicOrbitalAnsatz[str(v)] if not isinstance(v, AtomicOrbitalAnsatz) else v,
        metadata={MCTOOLS_METADATA_KEY: {
            'to_hdf5': lambda v: str(v.name),
        }},
    )

    integrals: dict[str, np.ndarray] = attrs.field(
        factory=dict,
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(str),
            value_validator=attrs.validators.instance_of(np.ndarray),
        ),
        repr=lambda integrals: f'{set(integrals)!r}',
        metadata={MCTOOLS_METADATA_KEY: {
            'hdf5_name': 'int1e'
        }},
    )

    # @df.validator
    # def _validate_df(self, attribute: attrs.Attribute, df: pd.DataFrame) -> None:
    #     for col in self.Col.required():
    #         assert col in df

    @integrals.validator
    def _validate_integrals(self, attribute: attrs.Attribute, integrals: dict[str, npt.NDArray]) -> None:
        for integral in integrals.values():
            assert integral.ndim == 2
            assert self.n_ao == integral.shape[0]

    def __attrs_post_init__(self):
        match self.ansatz:
            case AtomicOrbitalAnsatz.spherical:
                pass
            case _:
                raise NotImplementedError()

    @property
    def n_ao(self) -> int:
        return len(self.df)

    def get_metric(self) -> np.ndarray:
        return self.integrals['overlap']

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(#AO={self.n_ao}, ints={self.integrals})'

    # @classmethod
    # def from_resources(cls, resources: Resources) -> AtomicOrbitalBasis:
    #     df = cls.df_from_resources(resources)
    #     integrals = cls.integrals_from_resources(resources)
    #     return cls(df, integrals=integrals)

    @classmethod
    def df_from_resources(cls, resources: Resources) -> pd.DataFrame:
        df_dict = {
            cls.Col.shell_idx.value: resources.get(Resource.ao_basis_shell, None),
            cls.Col.atom_idx.value: resources.get(Resource.ao_basis_atom, None),
            cls.Col.l.value: resources.get(Resource.ao_basis_l, None),
            cls.Col.ml.value: resources.get(Resource.ao_basis_ml, None),
        }

        if any(v is None for v in df_dict.values()):
            raise ValueError('Cannot build atomic orbitals')

        if len(set(arr.shape for arr in df_dict.values())) > 1:
            raise ValueError('dimension mismatch')

        df = pd.DataFrame.from_dict(df_dict)

        atomic_numbers = resources[Resource.mol_atnums]
        df[cls.Col.element.value] = PeriodicTable.loc[atomic_numbers[df[cls.Col.atom_idx.value]], 'Symbol'].values
        df[cls.Col.L.value] = ANGULAR_MOMENTUM_SYMBS[df[cls.Col.l.value]]

        shell_coords = resources[Resource.ao_basis_shells_coords]
        df[cls.Col.get_xyz()] = shell_coords[df[cls.Col.shell_idx.value]]
        df[cls.Col.atom.value] = df[cls.Col.element.value] + df[cls.Col.atom_idx.value].astype(np.str_)

        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'ao_idx'
        return df

    @classmethod
    def integrals_from_resources(cls, resources: Resources) -> dict[str, np.ndarray]:
        integrals: dict[str, np.ndarray] = {}
        for r in Resource.STV():
            if (integral := resources.get(r)) is not None:
                integrals[r.name.split('_')[-1]] = integral
        return integrals

    def integrals_to_hdf5(self, filename: pathlib.Path, /, prefix: str = '') -> None:
        path, name = self.get_attr_hdf5_path('integrals', prefix=prefix)
        with h5py.File(filename, 'a') as file:
            gr = file.require_group('/'.join([path, name]))
            for array_name, array in self.integrals.items():
                gr.require_dataset(array_name, data=array, shape=array.shape, dtype=array.dtype)

    @classmethod
    def integrals_from_hdf5(cls, filename: pathlib.Path, /, prefix: str = '') -> dict[str, np.ndarray]:
        integrals: dict[str, np.ndarray] = {}
        path, name = cls.get_attr_hdf5_path('integrals', prefix=prefix)
        with h5py.File(filename, 'r') as file:
            gr = file.get('/'.join([path, name]))
            for array_name, ds in gr.items():
                integrals[array_name] = np.asarray(ds)
            del file
        return integrals

    # def to_hdf5(self, filename: pathlib.Path, /, prefix: str = '') -> None:
    #     with h5py.File(filename, 'a') as file:
    #         gr = file.require_group('/'.join([prefix, 'ao', 'basis']))
    #         gr.attrs['ansatz'] = str(self.ansatz.name)
    #
    #         gr = file.require_group('/'.join([prefix, 'ao', 'basis', 'df']))
    #         for col_name in self.df.columns:
    #             col = self.df[col_name]
    #             values = col.to_numpy(dtype=col.dtype if col.dtype != np.dtype(object) else np.dtype('S'))
    #             gr.require_dataset(col_name, data=values, dtype=values.dtype, shape=values.shape, compression='gzip')
    #
    #         gr = file.require_group('/'.join(['ao', 'int1e']))
    #         for name, array in self.integrals.items():
    #             gr.require_dataset(name, data=array, shape=array.shape, dtype=array.dtype, compression='gzip')

    # @classmethod
    # def from_hdf5(cls, filename: pathlib.Path, /, prefix: str = '') -> AtomicOrbitalBasis:
    #     with h5py.File(filename, 'r') as file:
    #         gr = file.require_group('/'.join([prefix, 'ao', 'basis']))
    #         ansatz = AtomicOrbitalAnsatz[gr.attrs['ansatz']]
    #
    #         if (gr := file.get('/'.join([prefix, 'ao', 'basis', 'df']), default=None)) is None:
    #             raise KeyError()
    #
    #         # TODO: Validate Columns
    #         df_dict = {}
    #         for col_name, col in gr.items():
    #             col = np.asarray(col)
    #             if col.dtype.kind == 'S':
    #                 col = col.astype('U')
    #             df_dict[col_name] = col
    #
    #         df = pd.DataFrame.from_dict(df_dict)
    #
    #         # TODO: Validate Integrals
    #         integrals = {}
    #         if (gr := file.get('/'.join(['ao', 'int1e']), None)) is None:
    #             raise KeyError()
    #
    #         for name, array in gr.items():
    #             integrals[name] = np.asarray(array)
    #
    #     return cls(df=df, ansatz=ansatz, integrals=integrals)

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
class MolecularOrbitalBasis(Consolidator):
    # RESOURCE_IDX_COL: ClassVar[str] = 'resource_idx'
    RESOURCE: ClassVar[Resource] = Resource.mo_basis
    ROOT: ClassVar[str] = '/'.join(['mo', 'basis'])

    class Col(StrEnum):
        occupied = auto()

        @classmethod
        def get_names(cls) -> list[str]:
            return [v.value for v in list(cls)]

        @classmethod
        def required(cls) -> list[str]:
            return [v.value for v in [cls.occupied.value]]

    # df: pd.DataFrame = attr.field(validator=attr.validators.instance_of(pd.DataFrame), repr=False)

    molorb: npt.NDArray = attrs.field(
        validator=attrs.validators.instance_of(np.ndarray),
        repr=False,
        metadata={MCTOOLS_METADATA_KEY: {
            'resource': Resource.mo_basis_molorb,
        }}
    )

    ansatz: MolecularOrbitalAnsatz = attrs.field(
        default=MolecularOrbitalAnsatz.GU,
        converter=lambda v: MolecularOrbitalAnsatz[str(v)] if not isinstance(v, MolecularOrbitalAnsatz) else v,
        metadata={MCTOOLS_METADATA_KEY: {
            'to_hdf5': lambda v: str(v.name),
            'resource': Resource.mo_basis_ansatz,
        }},
    )

    ao_basis: AtomicOrbitalBasis | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(AtomicOrbitalBasis)
        ),
        metadata={MCTOOLS_METADATA_KEY: {
            'resource': Resource.ao_basis
        }},
    )

    def __attrs_post_init__(self) -> None:
        assert self.n_mo == self.molorb.shape[0]

        match self.ansatz:
            case MolecularOrbitalAnsatz.GU:
                assert self.molorb.ndim == 3
                assert self.molorb.shape[-2] == 2
            case _:
                raise NotImplementedError()

        if self.ao_basis is not None:
            assert self.ao_basis.n_ao == self.molorb.shape[-1]

    @property
    def n_mo(self) -> int:
        return len(self.df)

    @property
    def n_occ(self) -> int:
        return sum(self.df[self.Col.occupied.value])

    @property
    def molorb_occ(self) -> np.ndarray:
        return self.molorb[:self.n_occ]

    @property
    def n_vir(self) -> int:
        return self.n_mo - self.n_occ

    @property
    def molorb_vir(self) -> np.ndarray:
        return self.molorb[self.n_vir:]

    def __len__(self) -> int:
        return len(self.df)

    def get_partitioning(self, /, by: list[str] = None, idx: slice | None = None, save: bool = False) -> pd.DataFrame:
        idx = idx or np.s_[:self.n_occ]
        by = by or [self.ao_basis.Col.atom.value, self.ao_basis.Col.L.value]

        S = self.get_metric()
        match self.ansatz:
            case MolecularOrbitalAnsatz.RR | MolecularOrbitalAnsatz.GU | MolecularOrbitalAnsatz.CR:
                C = self.molorb[idx]
            case MolecularOrbitalAnsatz.RU:
                raise NotImplementedError()

        F = []
        fragments = self.ao_basis.df[by].drop_duplicates().reset_index(drop=True)
        for i, fragment in fragments.iterrows():
            p = np.full(self.ao_basis.n_ao, True, dtype=np.bool_)
            for col, value in fragment.items():
                p &= self.ao_basis.df[col] == value
            F.append(np.outer(p, p) * S)
        F = np.stack(F)

        I = np.einsum('pat,Ftv,ab,pbv->Fp', C, F, I2, C.conj(), optimize='optimal')

        if save:
            labels = fragments[by].apply(lambda r: '_'.join(r[by]), axis=1)
            df = pd.DataFrame(I.T, columns=labels)
            self.df = pd.concat([self.df, df], axis=1)

        I = pd.DataFrame(I, columns=[f'MO{p}' for p in range(1, I.shape[-1] + 1)])
        return pd.concat([fragments, I], axis=1)

    def get_density(self, idx: slice | None = None) -> np.ndarray:
        idx = idx or np.s_[:self.n_occ]
        match self.ansatz:
            case MolecularOrbitalAnsatz.RR | MolecularOrbitalAnsatz.GU | MolecularOrbitalAnsatz.CR:
                C = self.molorb[idx]
            case MolecularOrbitalAnsatz.RU:
                raise NotImplementedError()

        return np.einsum('pat,qbv,ab->tv', C, C.conj(), I2, optimize='optimal')

    def get_overlap(self, idx: slice | None = None) -> np.ndarray:
        match self.ansatz:
            case MolecularOrbitalAnsatz.RR | MolecularOrbitalAnsatz.GU | MolecularOrbitalAnsatz.CR:
                C = self.molorb[idx]
            case MolecularOrbitalAnsatz.RU:
                raise NotImplementedError()

        S = self.get_metric()
        return np.einsum('pat,tv,qbv,ab->pq', C, S, C.conj(), I2, optimize='optimal')

    def get_metric(self) -> np.ndarray:
        return self.ao_basis.get_metric()

    # @classmethod
    # def from_resources(cls, storage: Resources) -> MolecularOrbitalBasis:
    #     df = cls.df_from_resources(storage)
    #
    #     molorb = storage[Resource.mo_basis_molorb]
    #     ansatz = storage[Resource.mo_basis_ansatz]
    #     ao_basis = storage[Resource.ao_basis]
    #
    #     return cls(df, molorb, ansatz, ao_basis=ao_basis)

    @classmethod
    def df_from_resources(cls, storage: Resources) -> pd.DataFrame:
        n_elec = storage[Resource.mol_nelec]
        n_mo = storage[Resource.mo_basis_molorb].shape[0]
        occupied = np.full(n_mo, False, dtype=np.bool_)
        occupied[:n_elec] = True

        df_dict = {
            cls.Col.occupied.name: occupied,
        }

        df = pd.DataFrame.from_dict(df_dict)

        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'mo_idx'
        return df

    # def to_hdf5(self, file: h5py.File, /, prefix: str = '') -> None:
    #     gr = file.require_group('/'.join([prefix, 'mo', 'basis']))
    #     gr.attrs['ansatz'] = str(self.ansatz.name)
    #
    #     gr = file.require_group('/'.join([prefix, 'mo', 'basis', 'df']))
    #     for col_name in self.df.columns:
    #         col = self.df[col_name]
    #         gr.require_dataset(col_name, data=col.values, dtype=col.dtype, shape=col.shape, compression='gzip')
    #
    #     file.require_dataset(
    #         '/'.join(['mo', 'molorb']),
    #         data=self.molorb,
    #         shape=self.molorb.shape,
    #         dtype=self.molorb.dtype,
    #         compression='gzip',
    #     )
    #
    #     self.ao_basis.to_hdf5(file, prefix=prefix)

    # @classmethod
    # def from_hdf5(cls, file: h5py.File, /, prefix: str = '') -> MolecularOrbitalBasis:
    #     gr = file.get('/'.join([prefix, 'mo', 'basis']))
    #     if gr is None:
    #         raise KeyError('No molecular orbital basis')
    #
    #     ansatz = MolecularOrbitalAnsatz[gr.attrs['ansatz']]
    #
    #     gr = file.require_group('/'.join([prefix, 'mo', 'basis', 'df']))
    #     if gr is None:
    #         raise KeyError('No molecular orbital df')
    #
    #     df_dict = {}
    #     for col_name, col in gr.items():
    #         col = np.asarray(col)
    #         if col.dtype.kind == 'S':
    #             col = col.astype('U')
    #         df_dict[col_name] = col
    #     df = pd.DataFrame.from_dict(df_dict)
    #
    #     ds = file.get('/'.join([prefix, 'mo', 'molorb']))
    #     if gr is None:
    #         raise KeyError('No molecular orbital coefficients')
    #
    #     molorb = np.asarray(ds)
    #     ao_basis = AtomicOrbitalBasis.from_hdf5(file, prefix=prefix)
    #
    #     return cls(df, molorb=molorb, ao_basis=ao_basis, ansatz=ansatz)

    @classmethod
    def get_build_resources(cls) -> Resource:
        return Resource.ao_basis | Resource.mol_nelec | Resource.mo_basis_molorb
