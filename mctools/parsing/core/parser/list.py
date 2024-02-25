from __future__ import annotations

import abc
from typing import TypeVar, TypeAlias, AnyStr
from typing_extensions import override

import attrs

from .base import Parser, FWP

__all__ = [
    'SequentialParser',
]

# FWP: TypeAlias = FileWithPosition[AnyStr]

R = TypeVar('R', covariant=True)  # Data from parse() type


@attrs.define(repr=True, eq=True)
class SequentialParser(Parser[list[R], AnyStr], metaclass=abc.ABCMeta):
    parsers: list[Parser[R, AnyStr]] = attrs.field(factory=list)
    storage: list[R | None] = attrs.field(factory=list)

    @override
    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[list[R | None], FWP[AnyStr]]:
        for parser in self.parsers:
            data, fwp = parser.parse(fwp)
            self.storage.append(data)
        return self.storage.copy(), fwp

    @override
    def cleanup(self, fwp: FWP[AnyStr]) -> FWP[AnyStr]:
        self.parsers.clear()
        self.storage.clear()
        return super(SequentialParser, self).cleanup(fwp)

    def add_parser(self, parser: Parser[R, AnyStr], /) -> None:
        self.parsers.append(parser)

    def __len__(self) -> int:
        return len(self.parsers)
