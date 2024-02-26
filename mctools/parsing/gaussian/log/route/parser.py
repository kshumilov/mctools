from __future__ import annotations

import re

from typing import AnyStr, ClassVar, TYPE_CHECKING, TypeAlias

import attrs

from ....core import Parser, LineStepper, ParsingError

from .route import Route, RouteLine
from ..utils import LINK_END_ANCHOR

if TYPE_CHECKING:
    from parsing.core.filehandler import FileWithPosition

    FWP: TypeAlias = FileWithPosition[AnyStr]


__all__ = [
    'RouteParser',
]


@attrs.define(repr=True, eq=True)
class RouteParser(Parser):
    RouteLinePattern: ClassVar[re.Pattern] = re.compile(
        r'^ *(\d+)/(\d+=[-\d]+,?)+/(\d,?)+(\([-\d+]\))?;'
    )

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    def parse_file(self, fwp: FWP[AnyStr]) -> tuple[Route, FWP[AnyStr]]:
        route = Route()

        self.stepper.take(fwp)
        until_predicate = self.stepper.get_anchor_predicate(LINK_END_ANCHOR)

        def parse_route_line(line: str | bytes) -> bool:
            if self.stepper.is_file_binary:
                line = line.decode()

            if '/' in line and self.RouteLinePattern.match(line):
                line = line.strip().strip(';')
                start, iops, ends, *step = list(
                    filter(None, re.split(r'[/()]', line))
                )

                start = int(start)

                iops = {
                    key: value
                    for key, value in map(
                        lambda p: tuple(map(int, p.split('='))), iops.split(','))

                }

                ends = [int(e) for e in ends.split(',')]
                match step:
                    case [step]:
                        step = int(step)
                    case []:
                        step = 0

                route_line = RouteLine(start, ends, iops, steps=step)
                route.append(route_line)

            return route.is_complete

        found = self.stepper.step_until(parse_route_line, until_predicate)
        if not found and not route.is_complete:
            raise ParsingError(f'Route is not complete: {route!r}')

        return route, self.stepper.return_file()

