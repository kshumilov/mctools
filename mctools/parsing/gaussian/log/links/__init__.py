from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, IO, AnyStr, Any, ClassVar, Iterable
from collections import deque, defaultdict

import attr

from parsing.core import (
    DispatchFileParser,
    Listeners,
    ParsingError, FileParser, SequentialParser, Parser,
)
from parsing.core.parser import ParserClassKey

from ..route import Link


if TYPE_CHECKING:
    from parsing.gaussian.log.route import Route
    from parsing.gaussian.log.links.parser import LinkParser


__all__ = [
    'LinksParser',
]


R = TypeVar('R')


@attr.define(repr=True)
class LinksParser(SequentialParser[R]):
    from .l302 import L302Parser

    DEFAULT_LINK_PARSERS: ClassVar[dict[Link, type[Parser]]] = {
        Link.L302: L302Parser,
    }

    @attr.define(repr=True)
    class Config(SequentialParser.Config):
        route: Route | None = attr.field(
            default=None,
            validator=attr.validators.optional([
                attr.validators.instance_of(Route),
            ])
        )

    def configure_parser(self, parser: LinkParser[R], config_key: ParserClassKey | int = 0) -> None:
        super(LinksParser, self).configure_parser(parser, config_key)

        match config_key:
            case int(idx):
                iops = self.config.route.links[parser][idx]
            case ParserClassKey(index=idx):
                iops = self.config.route.links[parser][idx]

        parser.update_config(iops=iops)

    def get_parser_classes(self) -> Iterable[type[LinkParser]]:
        for link, route_line in self.config.route.links:
            if link_parser_cls := self.DEFAULT_LINK_PARSERS.get(link):
                yield link_parser_cls

    # def __init__(self, route: Route, /, **kwargs) -> None:
    #     self.route = route
    #     if not len(route):
    #         raise ParsingError(f'{self!r}: Route is empty, check the log file: {route!r}')
    #
    #     super(LinksParser, self).__init__(**kwargs)

    # def build_listeners(self) -> Listeners:
    #     listeners = []
    #     for link, iops in self.route:
    #         if link_parser_cls := self.DEFAULT_LINK_PARSERS.get(link):
    #             link_parser = self.build_parser(link_parser_cls, iops)
    #             if link_parser.has_active_listeners:
    #                 listener = link_parser.get_link_listener()
    #                 listeners.append(listener)
    #             else:
    #                 print(f'{self!r}: Link parser {link_parser!r} is empty: skipping')
    #     return deque(listeners)


