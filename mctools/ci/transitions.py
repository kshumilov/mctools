from __future__ import annotations

from typing import ClassVar, Any

import attrs
import numpy as np
import pandas as pd

from mctools.newcore import Consolidator
from mctools.newcore.resource import Resource
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY
from .states import States

__all__ = [
    'Transitions'
]


@attrs.define(repr=True, eq=True)
class Transitions(Consolidator):
    RESOURCE: ClassVar[Resource] = Resource.ci_transitions
    ROOT = '/'.join(['ci/transitions'])

    tdms: np.ndarray = attrs.field(
        converter=np.asarray,
        validator=attrs.validators.instance_of(np.ndarray),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.ci_int1e_tdms}},
        repr=False
    )

    states: States = attrs.field(
        validator=attrs.validators.instance_of(States)
    )

    @classmethod
    def df_from_resources(cls, resources: dict[Resource, Any]) -> pd.DataFrame:
        df_dict: dict[str, np.ndarray] = {
            'idx': resources.get(Resource.ci_initial_idx),
            'fdx': resources.get(Resource.ci_final_idx),
            'osc': resources.get(Resource.ci_osc),
        }

        df = pd.DataFrame.from_dict(df_dict)
        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'state_idx'
        return df

