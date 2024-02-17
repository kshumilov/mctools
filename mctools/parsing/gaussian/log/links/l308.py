from __future__ import annotations

import re
from enum import Flag, auto
from typing import TextIO, Callable, TYPE_CHECKING

import numpy as np

from parsing.gaussian.log.route import Link
from parsing.gaussian.log.utils import read_matrix_in_file
from parsing import search_in_file

if TYPE_CHECKING:
    from parsing.gaussian.log.route import RouteLine
    from parsing.core.pattern import ParsingResultType

__all__ = [
    'read_rxdel_matrices',
    'l308_parser_funcs_general',
    'l308_postprocess_general',
]


class L308Properties(Flag):
    calculates_del_int = auto()
    calculates_rdel_int = auto()
    calculates_rdel_delr_int = auto()


rxp_header = re.compile(r'Calculate R x Del and Del integrals.')
rxp_int_header = re.compile(r'R x Del for IMat=\s*\d+:')


def read_rxdel_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResultType, str]:
    _, line = search_in_file(file, rxp_header, first_line=first_line,
                             err_msg='No r x del information is found')

    matrices = []
    for i in range(3):
        matrix, line = read_matrix_in_file(file, rxp_int_header, first_line=line)
        matrices.append(matrix)

    return {'rxdel': np.stack(matrices)}, line


def postprocess_velocity_integrals(result: ParsingResultType, integrals_key: str = 'integrals') -> None:
    integrals = result.setdefault(integrals_key, {})
    for integral in ['rxdel']:
        if integral in result:
            integrals[integral] = result.pop(integral)


def parse_l308_route(route_line: RouteLine) -> tuple[ParsingResultType, Flag] | None:
    if Link.L308.value not in route_line.links:
        return None

    calc_properties = L308Properties(0)

    value = route_line.get(36, 0)
    if value == -1:
        return {}, calc_properties

    calc_properties |= L308Properties.calculates_rdel_int
    match value // 100:
        case 0 | 1:
            calc_properties |= L308Properties.calculates_rdel_delr_int

    return {}, calc_properties


l308_parser_funcs_general: dict[str, list[Callable]] = {
    'l303': [
        read_rxdel_matrices,
    ],
}


l308_postprocess_general = {
    'l303': [
        postprocess_velocity_integrals,
    ]
}

