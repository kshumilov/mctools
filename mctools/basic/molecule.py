from __future__ import annotations

from typing import ClassVar, Sequence, Self

import attrs

from mctools.core.resource import Resource
from newcore.consolidator import Analyzer
from ..newcore.consolidator import Consolidator, Datum
from ..newcore.metadata import (
    MCTOOLS_METADATA_KEY,
    DFFieldMetadata,
    ResourcedFieldMetadata,
    DatumFieldMetadata,
    ConsolidatorFieldMetadata,
)

__all__ = [
    'Molecule',
]


@attrs.define
class Atom(Datum):
    Z: int = attrs.field(
        metadata={
            MCTOOLS_METADATA_KEY: DatumFieldMetadata(
                required=True,
                df_spec=DFFieldMetadata(
                    idx_axis=0,
                    value_axis=0,
                    cols=('Z',),
                ),
                res_spec=ResourcedFieldMetadata(
                    label=Resource.mol_atnums,
                )
            )
            # MCTOOLS_METADATA_KEY: {
            #     'required': True,  # Whether required at initialization
            #     'resource': {
            #         'label': Resource.mol_atnums,  # Where to find in the storage
            #         'ndim': 1,
            #     },
            #     'idx_axis': 0,  # Axis used to 'join' resource and the df atcoors
            #     'value_axis': 0,  # Axis used for values to be pulled into array
            #     'on_df': True,  # Whether stored on df or kept at the storage
            #     'col_name': 'Z',  # defaults to the attribute name
            # }
        }
    )

    coords: tuple[float, float, float] = attrs.field(
        metadata={
            MCTOOLS_METADATA_KEY: DatumFieldMetadata(
                required=True,
                df_spec=DFFieldMetadata(
                    idx_axis=0,
                    value_axis=1,
                    cols=('X', 'Y', 'Z'),
                ),
                res_spec=ResourcedFieldMetadata(
                    label=Resource.mol_atnums,
                    idx_axis=0,
                ),
            )
            # MCTOOLS_METADATA_KEY: {
            #     'required': True,  # Whether required at initialization
            #     'resource': {
            #         'label': Resource.mol_atnums,  # Where to find in the storage
            #         'ndim': 2,
            #     },
            #     'on_df': True,  # Whether stored on df or kept at the storage
            #     'idx_axis': 0,  # Axis used to 'join' resource and the df
            #     'value_axis': 1,  # Axis used for values to be pulled into array
            #     'col_name': ('X', 'Y', 'Z')
            # }
        }
    )

    elem: str = attrs.field(
        metadata={
            MCTOOLS_METADATA_KEY: DatumFieldMetadata(
                required=False,
                df_spec=DFFieldMetadata(
                    idx_axis=0,
                    value_axis=0,
                    cols=('El',),
                ),
            )
            # MCTOOLS_METADATA_KEY: {
            #     'resource': None,
            #     'required': False,
            #     'on_df': True,
            # }
        }
    )


@attrs.define
class Molecule(Consolidator):
    DatumClass: ClassVar[type[Atom]] = Atom
    ResourceLabel: ClassVar[Resource] = Resource.mol

    n_elec: int = attrs.field(
        converter=int,
        validator=attrs.validators.ge(1),
        metadata={
            MCTOOLS_METADATA_KEY: ConsolidatorFieldMetadata(
                required=True,
                res_spec=ResourcedFieldMetadata(
                    label=Resource.mol_nelec,
                    indexed=False
                )
            )
        }
    )

    def analyze(self, *analyzers: Sequence[Analyzer[Self]]) -> None:
        pass
