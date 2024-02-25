from __future__ import annotations

from typing import TypeAlias, AnyStr, Any, TYPE_CHECKING

import attrs

from core.resource import Resource

from parsing.core.parser import Parser
from parsing.core.filehandler import FileWithPosition
from parsing.gaussian.log.route import RouteParser
from parsing.gaussian.log.links import LinksParser

if TYPE_CHECKING:
    from parsing.gaussian.log.route import Route

    FWP: TypeAlias = FileWithPosition[AnyStr]
    D: TypeAlias = tuple[
        Route,
        dict[Resource, Any]
    ]
    R: TypeAlias = tuple[D, FWP]


@attrs.define(repr=True, eq=True)
class LogParser(Parser):
    requested: Resource = attrs.field(
        factory=Resource.ALL,
        validator=attrs.validators.instance_of(Resource),
    )
    filename: str = attrs.field(default='')

    def prepare(self, fwp: FWP[AnyStr], /) -> None:
        super(LogParser, self).prepare(fwp)
        self.filename = fwp.file.name

    def parse_file(self, fwp: FWP[AnyStr]) -> R[AnyStr]:
        route_parser = RouteParser()
        route, fwp = route_parser.parse(fwp)

        links_parser = LinksParser(route=route, requested=self.requested)
        links_data, fwp = links_parser.parse(fwp)
        return (route, links_data), fwp
