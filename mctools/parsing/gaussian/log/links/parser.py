from __future__ import annotations

from collections import defaultdict
from typing import ClassVar, Any, AnyStr, TypeAlias, TYPE_CHECKING

import attrs

from mctools.newcore.resource import Resource

from parsing.core.parser.base import Parser
from parsing.core.parser.list import SequentialParser
from parsing.gaussian.log.route.route import Route, Link

if TYPE_CHECKING:
    from parsing.gaussian.log.links.base import LinkParser
    from parsing.core.filehandler import FileWithPosition

    F: TypeAlias = FileWithPosition[AnyStr]


__all__ = [
    'LinksParser',
]


D: TypeAlias = list[dict[Resource, Any]]
R: TypeAlias = dict[Resource, list[Any]]


@attrs.define(repr=True, eq=True)
class LinksParser(Parser[R, AnyStr]):
    DEFAULT_LINK_PARSER_CLASSES: ClassVar[dict[Link, type[LinkParser]]] = {}

    requested: Resource = attrs.field(
        factory=Resource.ALL,
        validator=attrs.validators.instance_of(Resource),
    )

    include_links: defaultdict[Link, int] = attrs.field(
        factory=lambda: defaultdict(lambda: int(1)),
        converter=lambda v: defaultdict(lambda: int(1), v),
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.instance_of(Link),
            value_validator=attrs.validators.instance_of(int),
            mapping_validator=attrs.validators.instance_of(defaultdict)
        )
    )

    route: Route = attrs.field(default=attrs.Factory(Route))

    link_parsers: SequentialParser = attrs.field(factory=SequentialParser)
    links_parsed: list[Link] = attrs.field(factory=list, init=False)

    def is_ready(self) -> bool:
        return (self.route.is_complete and
                super(LinksParser, self).is_ready())

    def prepare(self, file: F, /) -> None:
        super(LinksParser, self).prepare(file)
        self.build_parsers()

    def build_parsers(self) -> None:
        for link, iops, available in self.route.get_available_resources():
            if link_parser_class := self.DEFAULT_LINK_PARSER_CLASSES.get(link):
                resources = Resource(available & link_parser_class.PARSABLE_RESOURCES)
                if resources != Resource.NONE() and self.include_links[link] > 0:
                    link_parser = link_parser_class(
                        resources=resources,
                        iops=iops
                    )
                    self.link_parsers.add_parser(link_parser)
                    self.links_parsed.append(link)
                    self.include_links[link] -= 1

    def parse_file(self, fwp: F, /) -> tuple[D, F]:
        return self.link_parsers.parse(fwp)

    def postprocess(self, raw_data: D) -> R:
        results = {}
        for link, link_result in zip(self.links_parsed, raw_data):
            for resource, result in link_result.items():
                results[resource] = result
        return super(LinksParser, self).postprocess(results)

    def __len__(self) -> int:
        return len(self.link_parsers)

    @classmethod
    def register(cls, link: Link, parser_class: type[LinkParser]) -> None:
        cls.DEFAULT_LINK_PARSER_CLASSES[link] = parser_class
