from __future__ import annotations

from typing import ClassVar

import attr

from core.resource import Resource

from parsing.core.parser.dict import Listener

from parsing.gaussian.log.links.base import LinkParser, RealMatrixParser

__all__ = [
    'L302Parser',
]


@attr.define(eq=True, repr=True)
class L302Parser(LinkParser):
    PARSABLE_RESOURCES: ClassVar[Resource] = Resource.STV()

    ANCHORS_STV = (
        'Overlap',
        'Kinetic',
        'Core Hamiltonian'
    )

    ANCHORS_X2C = (
        'Veff (p space)',
        'Trel (r space)',
        'Veff (r space)',
        'SO unc.',
        'DK / X2C integrals',
        'Orthogonalized basis functions'
    )

    DEFAULT_LISTENERS = {
        resource: Listener(
            parser=RealMatrixParser(),
            anchor=anchor,
            label=resource,
            max_runs=1
        )
        for resource, anchor in
        zip(Resource.STV(), ANCHORS_STV)
    }

    def postprocess(self, raw_data):
        data = {}
        for resource, resource_result in raw_data.items():
            if resource in list(self.PARSABLE_RESOURCES):
                data[resource] = resource_result.pop()
        return super().postprocess(data)
