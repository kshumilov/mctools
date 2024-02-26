from __future__ import annotations

from typing import TypeAlias, AnyStr, Any, TYPE_CHECKING, ClassVar

import attrs

from mctools.cli.console import console
from mctools.core.resource import Resource

from .route import Link, RouteParser
from ...core import Parser, SequentialParser
from ...core.filehandler import FileWithPosition

if TYPE_CHECKING:
    from parsing.gaussian.log.links.base import NewLinkParser
    from parsing.gaussian.log.route import Route

    FWP: TypeAlias = FileWithPosition[AnyStr]
    D: TypeAlias = dict[Resource, Any]
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

    def parse_file(self, fwp: FWP[AnyStr]) -> R[AnyStr]:
        console.rule(f'Log file: {fwp.file.name}')

        console.print("Parsing route...", )
        route_parser = RouteParser()
        route, fwp = route_parser.parse(fwp)

        links_parser = self.build_link_parsers(route)

        console.print("Parsing links...", new_line_start=True)
        links_data, fwp = links_parser.parse(fwp)

        result: dict[Resource, Any] = {}
        for link_data in links_data:
            result.update(link_data)

        return result, fwp

    def build_link_parsers(self, route):
        console.print('Found links: ', end=' ')
        parsers = SequentialParser()
        for link, iops, available in route.get_available_resources():
            if not (resources := self.requested & available):
                continue

            if (parser_class := self.LINK_PARSERS.get(link)) is None:
                continue

            console.print(f'{link.name}', end=' ')
            parser = parser_class(resources=resources, iops=iops)
            parsers.add_parser(parser)

        console.print()
        return parsers

    @classmethod
    def register_link_parser(cls, link: Link, parser_class: type[NewLinkParser]) -> None:
        cls.LINK_PARSERS[link] = parser_class


from .links import L302Parser, L910Parser

LogParser.register_link_parser(Link.L302, L302Parser)
LogParser.register_link_parser(Link.L910, L910Parser)
