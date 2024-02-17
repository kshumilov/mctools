from __future__ import annotations


from typing import TYPE_CHECKING, Any

from parsing.core import SequentialParser, ParsingError

from parsing.gaussian.log.route import Route, Link, RouteParser
from parsing.gaussian.log.links import LinksParser


if TYPE_CHECKING:
    from ...core.parser import BaseFileParserType

__all__ = [
    'LogParser',
    'RouteParser',
]


class LogParser(SequentialParser):
    Parsers = [
        RouteParser,
        LinksParser,
    ]

    def get_parser_args(self, parser_cls: type[BaseFileParserType]) -> list[Any]:
        args = super(LogParser, self).get_parser_args(parser_cls)

        if issubclass(parser_cls, LinksParser):
            for result in reversed(self.storage):
                if isinstance(result, Route):
                    args.append(result)
                    break
            else:
                raise ParsingError(f'{self!r}: Route not found')

        return args


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

