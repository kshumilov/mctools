from __future__ import annotations

from typing import ClassVar, Any

import attrs
import h5py
import numpy as np
import pandas as pd
import rich.repr
import scipy

from mctools.basic.basis import MolecularOrbitalBasis
from mctools.core.cistring import DASGraph
from mctools.newcore.consolidator import Consolidator
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY
from mctools.newcore.resource import Resource


@attrs.define(repr=True, eq=True)
class States(Consolidator):
    RESOURCE: ClassVar[Resource] = Resource.ci_states
    ROOT = '/'.join(['ci/states'])

    vecs: scipy.sparse.csr_matrix = attrs.field(
        validator=attrs.validators.instance_of(scipy.sparse.csr_matrix),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.ci_vecs}},
    )
    rdms: np.ndarray = attrs.field(
        converter=np.asarray,
        validator=attrs.validators.instance_of(np.ndarray),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.ci_int1e_rdms}},
        repr=False
    )
    graph: DASGraph = attrs.field(
        validator=attrs.validators.instance_of(DASGraph),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.ci_graph}},
    )
    mobasis: MolecularOrbitalBasis = attrs.field(
        validator=attrs.validators.instance_of(MolecularOrbitalBasis),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.mo_basis}},
        repr=False,
    )

    @property
    def n_active_mo(self) -> int:
        return self.graph.n_orb

    @property
    def n_inactive_mo(self) -> int:
        return self.mobasis.n_occ - self.n_active_mo

    @property
    def n_virtual_mo(self) -> int:
        return self.mobasis.n_mo - self.n_inactive_mo - self.graph.n_orb

    @property
    def actorb(self) -> np.ndarray:
        return self.mobasis.molorb[self.n_inactive_mo: self.n_inactive_mo + self.n_active_mo].copy()

    def __attrs_post_init__(self) -> None:
        self.mobasis.df['ci'] = pd.Categorical((['inactive'] * self.n_inactive_mo +
                                                ['active'] * self.n_active_mo +
                                                ['virtual'] * self.n_virtual_mo))

    @classmethod
    def df_from_resources(cls, resources: dict[Resource, Any]) -> pd.DataFrame:
        df_dict: dict[str, np.ndarray] = {
            'idx': resources.get(Resource.ci_state_idx),
            'E': resources.get(Resource.ci_energies),
        }

        df = pd.DataFrame.from_dict(df_dict)
        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'state_idx'
        return df

    @classmethod
    def get_build_resources(cls) -> Resource:
        return (Resource.ci_energies | Resource.ci_int1e_rdms |
                Resource.ci_graph | Resource.ci_state_idx | Resource.mo_basis)

    def vecs_to_hdf5(self, filename: str, /, prefix: str = '') -> None:
        path, name = self.get_attr_hdf5_path('vecs', prefix=prefix)
        with h5py.File(filename, 'a') as file:
            gr = file.require_group('/'.join([path, name]))
            gr.attrs['shape'] = self.vecs.shape
            for arr_name in ['data', 'indices', 'indptr']:
                array = getattr(self.vecs, arr_name)
                gr.require_dataset(arr_name, data=array, shape=array.shape, dtype=array.dtype)

    @classmethod
    def vecs_from_hdf5(cls, filename: str, /, prefix: str = '') -> scipy.sparse.csr_matrix:
        path, name = cls.get_attr_hdf5_path('vecs', prefix=prefix)
        with h5py.File(filename, 'r') as file:
            gr = file.get('/'.join([path, name]))
            args = []
            for arr_name in ['data', 'indices', 'indptr']:
                ds = gr.get(arr_name)
                args.append(np.asarray(ds))
            shape = tuple(gr.attrs['shape'])
        del file, gr, ds
        return scipy.sparse.csr_matrix(tuple(args), shape=shape)
