from __future__ import annotations

import abc
from typing import TypeVar, TypeAlias, AnyStr, TYPE_CHECKING
from typing_extensions import override

import attrs

from .base import Parser

__all__ = [
    'SequentialParser',
]

if TYPE_CHECKING:
    from ..filehandler import FileWithPosition

    FWP: TypeAlias = FileWithPosition[AnyStr]


R = TypeVar('R', covariant=True)  # Data from parse() type


@attrs.define(repr=True, eq=True)
class SequentialParser(Parser[list[R], AnyStr], metaclass=abc.ABCMeta):
    parsers: list[tuple[Parser[R, AnyStr], int]] = attrs.field(factory=list)

    @parsers.validator
    def _validate_parser(self, attribute, parsers):
        for item in parsers:
            if len(item) != 2:
                raise ValueError()

            parser, n = item
            if not isinstance(parser, Parser) or not isinstance(n, int) or n < 0:
                raise ValueError()

    @override
    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[list[R | None], FWP[AnyStr]]:
        result: list[R | None] = []
        for parser, n in self.parsers:
            for _ in range(n):
                data, fwp = parser.parse(fwp)
                result.append(data)
        return result, fwp

    def add_parser(self, parser: Parser[R, AnyStr], /, n: int = 1) -> None:
        self.parsers.append((parser, n))

    def __len__(self) -> int:
        return len(self.parsers)
