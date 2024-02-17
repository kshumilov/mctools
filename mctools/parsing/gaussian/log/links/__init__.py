from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque

from parsing.core import (
    DispatchFileParser,
    CompositeParser,
    Listeners,
    ParsingError,
)

from parsing.gaussian.log.route import Link


if TYPE_CHECKING:
    from parsing.gaussian.log.route import Route
    from .parser import LinkParserType


__all__ = [
    'LinksParser',
]


class LinksParser(DispatchFileParser, CompositeParser):
    from .l302 import L302Parser

    link_parsers: dict[Link, type[LinkParserType]] = {
        Link.L302: L302Parser,
    }

    def __init__(self, route: Route, /, **kwargs) -> None:
        self.route = route
        if not len(route):
            raise ParsingError(f'{self!r}: Route is empty, check the log file: {route!r}')

        super(LinksParser, self).__init__(**kwargs)

    def build_listeners(self) -> Listeners:
        listeners = []
        for link, iops in self.route:
            if link_parser_cls := self.link_parsers.get(link):
                link_parser = self.build_parser(link_parser_cls, iops)
                if link_parser.has_active_listeners:
                    listener = link_parser.get_link_listener()
                    listeners.append(listener)
                else:
                    print(f'{self!r}: Link parser {link_parser!r} is empty: skipping')
        return deque(listeners)


