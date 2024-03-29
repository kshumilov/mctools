from __future__ import annotations

from typing import ClassVar, Any, Sequence

import attrs
import numpy as np
import pandas as pd

from mctools.newcore import Consolidator
from mctools.newcore.resource import Resource
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY

from .states import States
from .common import CI_ROOT

__all__ = [
    'Transitions',
]


@attrs.define(repr=True, eq=True)
class Transitions(Consolidator):
    RESOURCE: ClassVar[Resource] = Resource.ci_transitions
    ROOT = '/'.join([CI_ROOT, 'transitions'])

    states: States = attrs.field(
        validator=attrs.validators.instance_of(States)
    )

    tdms: np.ndarray | None = attrs.field(
        default=None,
        converter=np.asarray,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(np.ndarray)
        ),
        metadata={MCTOOLS_METADATA_KEY: {'resource': Resource.ci_int1e_tdms}},
        repr=False
    )

    def get_state_properties(self, props: Sequence[str]) -> pd.DataFrame:
        dfs = self.states.df[['idx', *props]]

        # df = ((self.df[['idx', 'fdx']].merge(
        #     dfs, how='left',
        #     right_on=['idx'],
        #     left_on=['idx'],
        #     copy=True
        # ).drop(columns=['idx']).rename(
        #     columns={col: f'{col}_i' for col in props}
        # ).merge(
        #     dfs, how='left',
        #     right_on=['idx'],
        #     left_on=['fdx'],
        #     copy=True
        # ).drop(columns=['fdx']).rename(
        #     columns={col: f'{col}_f' for col in props}
        # )))

        df = (
            self.df[['idx', 'fdx']]
            .merge(dfs, how='left', right_on=['idx'], left_on=['idx'])
            .drop(columns=['idx'])
            .rename(columns={col: f'{col}_i' for col in props})
            .merge(dfs, how='left', left_on=['fdx'], right_on=['idx'])
            .rename(columns={col: f'{col}_f' for col in props})
            .drop(columns=['idx', 'fdx'])
        )

        return df

    @classmethod
    def df_from_resources(cls, resources: dict[Resource, Any]) -> pd.DataFrame:
        df_dict: dict[str, np.ndarray] = {
            'idx': resources.get(Resource.ci_initial_idx),
            'fdx': resources.get(Resource.ci_final_idx),
            'osc': resources.get(Resource.ci_osc),
        }

        df = pd.DataFrame.from_dict(df_dict)
        df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
        df.index.name = 'transition_idx'
        return df

    @classmethod
    def get_build_resources(cls) -> Resource:
        return (
            Resource.ci_states |
            Resource.ci_initial_idx |
            Resource.ci_final_idx |
            Resource.ci_osc
        )
