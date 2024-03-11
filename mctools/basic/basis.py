from __future__ import annotations

import pathlib
from enum import unique, StrEnum, auto
from typing import TypeAlias, Any, ClassVar, Hashable
from collections import defaultdict

import attrs
import h5py
import numpy as np
import pandas as pd
import rich.repr
from numpy import typing as npt

from mctools.core.utils.constants import PeriodicTable, ANGULAR_MOMENTUM_SYMBS
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
    spinor = auto()
    spherical = auto()
    cartesian = auto()


def transform_integral(integral: npt.NDArray, transformation: npt.NDArray) -> npt.NDArray:
    return np.einsum('pi,...ij,qj->...pq', transformation, integral, transformation.conj(), optimize='optimal')


Resources: TypeAlias = dict[Resource, Any]


@attrs.define(repr=True, eq=True)
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

    def __attrs_post_init__(self) -> None:
        match self.ansatz:
            case AtomicOrbitalAnsatz.cartesian:
                raise NotImplementedError()

    @property
    def n_ao(self) -> int:
        return len(self.df)

    def get_metric(self) -> np.ndarray:
        return self.integrals['overlap']

    def __len__(self) -> int:
        return len(self.df)

    def __rich_repr__(self) -> rich.repr.Result:
        yield '#AO', self.n_ao
        yield 'ansatz', self.ansatz.value
        yield 'integrals', set(self.integrals)

    def get_fragments(self, cols: list[str], /) -> pd.DataFrame:
        fragments: defaultdict[Hashable, np.ndarray] = defaultdict(
            lambda: np.full(self.n_ao, False, dtype=np.bool_)
        )

        for label, ao_idx in self.df.groupby(cols).indices.items():
            fragments[label][ao_idx] = True

        if len(cols) > 1:
            index = pd.MultiIndex.from_tuples(fragments.keys(), names=cols)
        else:
            index = pd.Index(fragments.keys(), name=cols[0])

        return pd.DataFrame(
            fragments.values(),
            index=index,
            columns=[f'AO{i + 1}' for i in range(self.n_ao)]
        )

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

        n_prim = resources.get(Resource.ao_basis_shells_size, None)
        df['n_prim'] = n_prim[df.shell_idx]

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

    @classmethod
    def get_build_resources(cls) -> Resource:
        return (Resource.ao_basis_shell | Resource.ao_basis_atom | Resource.ao_basis_l | Resource.ao_basis_ml |
                Resource.ao_basis_shells_coords | Resource.ao_basis_shells_size | Resource.ao_basis_prims_coef |
                Resource.ao_basis_prims_exp |
                Resource.mol_atnums | Resource.mol_atnums)


@unique
class MolecularOrbitalAnsatz(StrEnum):
    RR = 'Real Restricted'
    CR = 'Complex Restricted'
    RU = 'Real Unrestricted'
    GU = 'Complex Unrestricted'


@attrs.define(repr=True, eq=True)
class MolecularOrbitalBasis(Consolidator):
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

    def __rich_repr__(self) -> rich.repr.Result:
        yield "#MO", dict(Occ=self.n_occ, Vir=self.n_vir)
        yield "#Occ", self.n_occ,
        yield "ansatz", self.ansatz.value
        yield self.ao_basis

    def get_partition(self, by: list[str], /) -> pd.DataFrame:
        fragments = self.ao_basis.get_fragments(by)

        S = self.ao_basis.get_metric()

        overlap = np.einsum(
            'Ft,pat,tu,ab,pbu -> Fp',
            fragments.to_numpy(),
            self.molorb, S, np.eye(2), self.molorb.conj(),
            optimize=True
        )

        return pd.DataFrame(
            overlap,
            columns=[f'MO{i + 1}' for i in range(self.n_mo)],
            index=fragments.index,
        )

    def get_population(self, of: list[str], /) -> pd.DataFrame:
        fragments = self.ao_basis.get_fragments(of)

        S = self.ao_basis.get_metric()
        P = self.get_density()

        Q: np.ndarray = np.einsum(
            'abtu,ba,ut -> t',
            P, np.eye(2), S,
            optimize=True
        )

        return (fragments * Q).sum(axis=1)

    def get_density(self) -> np.ndarray:
        C = self.molorb[:self.n_occ]

        return np.einsum(
            'pat,pbu->abtu',
            C, C.conj(),
            optimize=True,
        )

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

    @classmethod
    def get_build_resources(cls) -> Resource:
        return Resource.ao_basis | Resource.mol_nelec | Resource.mo_basis_molorb

    @classmethod
    def from_other_idx(cls, other: MolecularOrbitalBasis, idx: slice) -> MolecularOrbitalBasis:
        return cls(
            df=other.df.iloc[idx].copy(),
            molorb=other.molorb[idx].copy(),
            ansatz=other.ansatz,
            ao_basis=other.ao_basis,
        )
