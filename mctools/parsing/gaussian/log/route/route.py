from __future__ import annotations

import pathlib
import re
from collections import defaultdict

from enum import unique, StrEnum, auto
from typing import TypeAlias, MutableMapping, Iterator, Sequence

import attrs

from core.resource import Resource


@unique
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


IOps: TypeAlias = dict[int, int]


@attrs.define(repr=True, eq=True)
class RouteLine(MutableMapping[int, int]):
    start: int = attrs.field(validator=[attrs.validators.instance_of(int), attrs.validators.ge(0)])
    ends: list[int] = attrs.field(validator=attrs.validators.deep_iterable(
        member_validator=[
            attrs.validators.instance_of(int),
            attrs.validators.ge(0),
        ],
    ))
    iops: IOps = attrs.field(
        factory=lambda: defaultdict(int),
        converter=lambda d: defaultdict(int, d),
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.ge(0),
            value_validator=attrs.validators.instance_of(int),
            mapping_validator=attrs.validators.instance_of(defaultdict)))
    steps: int = attrs.field(default=0, validator=attrs.validators.instance_of(int))

    def __getitem__(self, iop: int) -> int:
        return self.iops[iop]

    def __setitem__(self, iop: int, value: int) -> None:
        if iop <= 0:
            raise ValueError(f"{self!r}: IOp key must all be positive: {iop}")

        self.iops[iop] = value

    def __delitem__(self, iop: int) -> None:
        del self.iops[iop]

    def __iter__(self) -> Iterator[int]:
        return iter(self.iops)

    def __len__(self) -> int:
        return len(self.iops)

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
    def from_route_line(cls, line: str) -> RouteLine | None:
        if cls.is_valid_route_line(line):
            return cls.parse_route_line(line)

    @classmethod
    def parse_route_line(cls, line: str) -> RouteLine | None:
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


@attrs.define(repr=True, eq=True, slots=True)
class Route:
    route_lines: list[RouteLine] = attrs.field(
        default=attrs.Factory(list), init=True,
        validator=attrs.validators.deep_iterable(
            member_validator=attrs.validators.instance_of(RouteLine),
            iterable_validator=attrs.validators.instance_of(list),
        ),
        repr=False,
    )

    links: defaultdict[Link, list[RouteLine]] = attrs.field(
        factory=lambda: defaultdict(list),
        init=False,
        repr=lambda links: ','.join(links.keys())
    )

    def __attrs_post_init__(self):
        for route_line in self.route_lines:
            for link in route_line.links:
                self.links[link].append(route_line)

    def __len__(self) -> int:
        return len(self.links)

    def __iter__(self) -> Iterator[tuple[Link, dict[int, int]]]:
        for link, route_lines in self.links.items():
            for route_line in route_lines:
                yield link, route_line.iops

    def __contains__(self, link: Link) -> bool:
        return link in self.links

    def __str__(self) -> str:
        return '\n'.join(map(str, self.route_lines)) + '\n'

    def append(self, route_line: RouteLine) -> None:
        self.route_lines.append(route_line)
        for link in route_line.links:
            self.links[link].append(route_line)

    @property
    def is_complete(self) -> bool:
        return Link.L9999 in self.links

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

    def get_available_resources(self) -> Sequence[tuple[Link, IOps, Resource]]:
        available: list[tuple[Link, IOps, Resource]] = []
        for link, iops in self:
            link_resources = self.get_link_resources(link, iops)
            available.append((link, iops, link_resources))
        return available

    def get_link_resources(self, link: Link, iops: IOps) -> Resource:
        if link not in self.links:
            return Resource.NONE()

        func_name = f'{link.value}_resources'
        if not hasattr(self, func_name):
            return Resource.NONE()

        return getattr(self, func_name)(iops)

    @staticmethod
    def l302_resources(iops: IOps) -> Resource:
        if iops[33] in (1, 5):
            return Resource.STV()
        return Resource.NONE()

    @staticmethod
    def l910_resources(iops: IOps) -> Resource:
        resources = Resource.NONE()

        if iops[17] > 0:
            resources |= Resource.ci_energy

        if iops[18] > 0:
            resources |= Resource.ci_saweights

        if iops[19] > 0:
            resources |= Resource.ci_osc

        if iops[31] in [0, 1]:
            resources |= Resource.ci_int1e_rdms

        if iops[54] > 0:
            resources |= Resource.ci_spin

        if iops[133] > 0:
            resources |= Resource.ci_int1e_tdms

        return Resource(resources)
