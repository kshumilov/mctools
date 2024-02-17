from __future__ import annotations

import re
import pathlib

from enum import StrEnum, auto
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence, Iterator, MutableMapping, ClassVar, AnyStr, IO, Any

from parsing.core import FileParser, ParsingError

__all__ = [
    'Link',
    'RouteLine',
    'Route',
    'RouteParser',

    'IOps',
]


class Link(StrEnum):
    L101 = auto()
    L202 = auto()
    L301 = auto()
    L302 = auto()
    L303 = auto()
    L308 = auto()
    L401 = auto()
    L910 = auto()
    L601 = auto()
    L9999 = auto()


IOps = dict[int, int]


@dataclass(slots=True, eq=True)
class RouteLine(MutableMapping[int, int]):
    RouteLinePattern: ClassVar[re.Pattern] = re.compile(r'^ *(\d+)/(\d+=[-\d]+,?)+/(\d,?)+(\([-\d+]\))?;')

    start: int
    ends: list[int] = field(default_factory=list)
    iops: IOps = field(default_factory=dict)
    steps: int = 0

    def __post_init__(self):
        if self.start <= 0:
            raise ValueError(f"{self!r}: 'start' must be greater than 0: {self.start}")

        if any(e <= 0 for e in self.ends):
            raise ValueError(f"{self!r}: 'ends' must all all greater than 0: {self.ends}")

        if any(p <= 0 for p in self.iops.keys()):
            raise ValueError(f"{self!r}: parameter keys must all be greater than 0: {self.iops.keys()}")

    def __getitem__(self: 'RouteLine', param: int) -> int:
        return self.iops[param]

    def __setitem__(self, iop: int, value: int) -> None:
        if iop <= 0:
            raise ValueError(f"{self!r}: IOp key must all be positive: {iop}")

        self.iops[iop] = value

    def __delitem__(self, iop: int) -> None:
        del self.iops[iop]

    def __iter__(self: 'RouteLine') -> Iterator[int]:
        return iter(self.iops)

    def __len__(self: 'RouteLine') -> int:
        return len(self.iops)

    def __repr__(self) -> str:
        links = ','.join(self.links)
        return f'{self.__class__.__name__}(links=[{links}], iops={self.iops!r})'

    def __str__(self) -> str:
        iops = ','.join(
            '='.join(map(str, iop))
            for iop in self.iops.items()
        )
        ends = ','.join(map(str, self.ends))
        route_line = '/'.join([str(self.start), iops, ends])
        steps = f'({self.steps!s})' if self.steps else ''
        return f'{route_line}{steps};'

    @property
    def links(self) -> tuple[Link, ...]:
        return tuple([Link(f'l{self.start * 100 + end}') for end in self.ends])

    @classmethod
    def is_valid_route_line(cls, line: str) -> bool:
        return cls.RouteLinePattern.match(line) is not None

    @classmethod
    def from_route_line(cls, line: str) -> Optional['RouteLine']:
        if cls.is_valid_route_line(line):
            return cls.parse_route_line(line)

    @classmethod
    def parse_route_line(cls, line: str) -> 'RouteLine':
        line = line.strip().strip(';')

        start: int
        start, params, ends, *step = list(filter(None, re.split(r'[/()]', line)))

        start = int(start)
        params = {k: v for k, v in map(lambda p: tuple(map(int, p.split('='))), params.split(','))}
        ends = [int(e) for e in ends.split(',')]
        match step:
            case [step]:
                step = int(step)
            case []:
                step = 0

        return RouteLine(start=start, ends=ends, iops=params, steps=step)


class Route:
    __slots__ = (
        'route_lines',
        'links',
    )

    route_lines: list[RouteLine]
    links: dict[Link, list[RouteLine]]

    def __init__(self, route_lines: Sequence[RouteLine] | None = None) -> None:
        self.route_lines = route_lines if route_lines is not None else []

        self.links = defaultdict(list)
        for route_line in self.route_lines:
            for link in route_line.links:
                self.links[link].append(route_line)

    def __len__(self: 'Route') -> int:
        return len(self.links)

    def __iter__(self) -> Iterator[tuple[Link, dict[int, int]]]:
        for link, route_lines in self.links.items():
            for route_line in route_lines:
                yield link, route_line.iops

    def __contains__(self, link: Link) -> bool:
        return link in self.links

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(links=[%s])' % ','.join(self.links.keys())

    def __str__(self) -> str:
        return '\n'.join(map(str, self.route_lines)) + '\n'

    @property
    def is_complete(self) -> bool:
        return Link.L9999 in self.links

    def append(self, route_line: RouteLine) -> None:
        self.route_lines.append(route_line)
        for link in route_line.links:
            self.links[link].append(route_line)

    @classmethod
    def from_log(cls, filename: str | pathlib.Path) -> 'Route':
        logfile = pathlib.Path(filename)
        with logfile.open(mode="r") as file:
            route = cls()
            for line in file:
                if line.startswith(' Leave'):
                    break

                if route_line := RouteLine.from_route_line(line):
                    route.append(route_line)

        return route


class RouteParser(FileParser):
    def _parse_file(self, file: IO[AnyStr]) -> Any:
        super(RouteParser, self)._parse_file(file)

        route = Route()

        def add_route_line(line: AnyStr) -> bool:
            if ('/' in line and
                    (route_line := RouteLine.from_route_line(line)) is not None):
                route.append(route_line)
                if route.is_complete:
                    return True
            return False

        until_predicate = self.stepper.get_predicate('Leave Link')
        self.stepper.step_to_first(add_route_line, until_predicate)

        if not route.is_complete:
            raise ParsingError(f'{self!r}: Route is not complete: {route!r}')
        return route
