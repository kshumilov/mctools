from __future__ import annotations

import pathlib
from collections import defaultdict, Counter
from typing import TypeAlias, AnyStr, Any, TYPE_CHECKING, ClassVar

import attrs
import h5py
import numpy as np

from core.resource import Resource

from parsing.core import Parser, SequentialParser
from parsing.core.filehandler import FileWithPosition
from parsing.gaussian.log.route import Link, RouteParser

# from parsing.gaussian.log.links import LinksParser

if TYPE_CHECKING:
    from parsing.gaussian.log.links.base import NewLinkParser
    from parsing.gaussian.log.route import Route

    FWP: TypeAlias = FileWithPosition[AnyStr]
    D: TypeAlias = tuple[
        Route,
        dict[Resource, Any]
    ]
    R: TypeAlias = tuple[D, FWP]


__all__ = [
    'LogParser'
]


@attrs.define(repr=True, eq=True)
class LogParser(Parser):
    LINK_PARSERS: ClassVar[dict[Link, type[NewLinkParser]]] = {}

    requested: Resource = attrs.field(
        factory=Resource.ALL,
        validator=attrs.validators.instance_of(Resource),
    )

    save: bool = attrs.field(default=True)

    def parse_file(self, fwp: FWP[AnyStr]) -> R[AnyStr]:
        route_parser = RouteParser()
        route, fwp = route_parser.parse(fwp)

        links_parser = self.build_link_parsers(route)
        links_data, fwp = links_parser.parse(fwp)

        result: dict[Resource, Any] = {}
        for link_data in links_data:
            result.update(link_data)

        if self.save:
            self.save_data(result, fwp)

        return (route, result), fwp

    def build_link_parsers(self, route):
        parsers = SequentialParser()
        for link, iops, available in route.get_available_resources():
            if not (resources := self.requested & available):
                continue

            if (parser_class := self.LINK_PARSERS.get(link)) is None:
                continue

            parser = parser_class(resources=resources, iops=iops)
            parsers.add_parser(parser)
        return parsers

    @classmethod
    def register_link_parser(cls, link: Link, parser_class: type[NewLinkParser]) -> None:
        cls.LINK_PARSERS[link] = parser_class

    def save_data(self, result: dict[Resource, np.ndarray], fwp: FWP[AnyStr]) -> None:
        filename = pathlib.Path(fwp.file.name).with_suffix('.h5')
        with h5py.File(filename, libver='latest') as f:
            for label, resource in result.items():
                if isinstance(resource, np.ndarray):
                    name = '/'.join(label.name.split('_'))
                    f.create_dataset(
                        name, data=resource, dtype=resource.dtype,
                        compression='gzip'
                    )


from .links import L302Parser, L910Parser

LogParser.register_link_parser(Link.L302, L302Parser)
LogParser.register_link_parser(Link.L910, L910Parser)
