from __future__ import annotations

from typing import Any

import attrs
import h5py

from mctools.core.resource import Resource
from mctools.cli.console import console

__all__ = [
    'Storage',
]


@attrs.define(repr=True, eq=True)
class Storage:
    resources: dict[Resource, Any] = attrs.field(factory=dict)

    consolidators: list[type] = attrs.field(factory=list)
    complete: list[Any] = attrs.field(factory=list)

    def __attrs_post_init__(self) -> None:
        self.build_consolidators()

    def add_resource(self, resource: Resource, data: Any) -> None:
        self.resources[resource] = data
        self.build_consolidators()

    def build_consolidators(self) -> None:
        to_pop: list[int] = []

        results = []
        available = self.get_available_resources()
        for i, builder in enumerate(self.consolidators):
            required = builder.get_build_resources()
            if len(available & required) == len(required):
                result = builder.from_resources(self.resources)
                results.append((builder.RESOURCE, result))
                self.complete.append(result)
                to_pop.append(i)

        for i in to_pop:
            self.consolidators.pop(i)

        for resource, result in results:
            self.add_resource(resource, result)

    def get_available_resources(self) -> Resource:
        available = Resource.NONE()
        for resource in self.resources.keys():
            available |= resource
        return available

    def to_hdf5(self, file: h5py.File, /, prefix: str = '') -> None:
        for consolidator in self.complete:
            consolidator.to_hdf5(file, prefix=prefix)
