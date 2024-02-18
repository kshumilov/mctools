from __future__ import annotations


from typing import TYPE_CHECKING, Any, IO, AnyStr

import attr

from parsing.core.parser import SequentialParser, ParserClassKey

from parsing.gaussian.log.route import Route, Link, RouteParser
from parsing.gaussian.log.links import LinksParser

if TYPE_CHECKING:
    from parsing.core import Parser


__all__ = [
    'LogParser',
    'RouteParser',
]


class LogParser(SequentialParser):
    Parsers = [
        RouteParser,
        LinksParser,
    ]

    def update_config(self, parser: Parser, data, config_key: ParserClassKey) -> None:
        if isinstance(parser, RouteParser):
            links_config = self.config.parsers_config[LinksParser]
            links_config = attr.evolve(links_config, route=data)
            self.config.parsers_config[LinksParser] = links_config


if __name__ == '__main__':
    import pathlib
    data_dir = pathlib.Path('/Users/kirill/Documents/UW/Li_Group/mctools/examples').resolve()
    calc_dir = data_dir / 'calc'
    log = calc_dir / 'casscf.log'
    fchk = log.with_suffix('.fchk')

    from mctools.core import Resources
    parser = LogParser()
    data = parser.parse(log)
    print(data)

