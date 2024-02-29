from __future__ import annotations

from typing import ClassVar

import attrs
import numpy as np

from mctools.newcore import Consolidator
from mctools.newcore.resource import Resource
from mctools.newcore.metadata import MCTOOLS_METADATA_KEY

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

    states: States = attrs.field()


