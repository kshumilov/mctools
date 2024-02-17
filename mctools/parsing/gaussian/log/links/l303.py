from __future__ import annotations

import re

from enum import Flag, auto
from itertools import permutations
from collections import defaultdict
from typing import TextIO, Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from parsing.gaussian.log.route import Link
from parsing.gaussian.log.utils import read_matrix_in_file
from parsing.core import ProcessedPattern, DIMS_MAP
from parsing import search_in_file

if TYPE_CHECKING:
    from parsing.gaussian.log.route import RouteLine
    from parsing.core.pattern import ParsingResultType

__all__ = [
    'MULTIPOLE_INTEGRALS',

    'read_multipole_matrices',
    'l303_parser_funcs_general',
    'l303_postprocess_general',
]


class L303Properties(Flag):
    calculates_multipole_ints = auto()


MULTIPOLE_INTEGRALS = ['dipole', 'quadrupole', 'octupole', 'hexapole']
MULTIPOLE_SIZES = {name: i for i, name in enumerate(MULTIPOLE_INTEGRALS, 1)}

MULTIPOLE_INTEGRAL_LABELS = np.asarray([
    tuple(map(lambda c: DIMS_MAP[c], char.strip()))
    for char in 'X,Y,Z,XX,YY,ZZ,XY,XZ,YZ,XXX,YYY,ZZZ,XYY,XXY,XXZ,XZZ,YZZ,YYZ,XYZ,'
                'XXXX,YYYY,ZZZZ,XXXY,XXXZ,YYYX,YYYZ,ZZZX,ZZZY,XXYY,XXZZ,YYZZ,XXYZ,YYXZ,ZZXY'.lower().split(',')
], dtype=object)


multipole_header = ProcessedPattern(
    r'Multipole integrals L=\d+ to (?P<max_m>\d+) MinM= 0 MaxM=(?P<N>\d+).',
    'multipole_info', default_group_map=int
)

# 'Multipole matrices IBuc=   518 IX=    1 IJ=           1:'
multipole_int_header = re.compile(r'Multipole matrices ')  # Appears 3 times


def read_multipole_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResultType, str]:
    info, line = search_in_file(file, multipole_header,
                                first_line=first_line,
                                err_msg='No Multipole information is found')

    multipoles: dict[str, dict[tuple[int, ...], npt.NDArray] | npt.NDArray] = defaultdict(dict)
    for _, int_label in zip(range(info.N), MULTIPOLE_INTEGRAL_LABELS):
        multipole_label = MULTIPOLE_INTEGRALS[len(int_label) - 1]

        matrix, line = read_matrix_in_file(file, multipole_int_header, first_line=line)
        multipoles[multipole_label][int_label] = matrix

        n_ao = matrix.shape[-1]

    result = {}
    while multipoles:
        multipole_label, integrals = multipoles.popitem()

        size = MULTIPOLE_SIZES[multipole_label]
        multipole = np.zeros((3,) * size + (n_ao,) * 2)
        for int_label, matrix in integrals.items():
            for p in set(permutations(int_label)):
                multipole[p] = matrix

        result[multipole_label] = multipole

    return result, line


def postprocess_multipole_integrals(result: ParsingResultType, integrals_key: str = 'integrals') -> None:
    integrals = result.setdefault(integrals_key, {})
    for integral in MULTIPOLE_INTEGRALS + ['rxdel']:
        if integral in result:
            integrals[integral] = result.pop(integral)


fermi_header = re.compile(r'Fermi contact integrals:')  # Not square


def parse_l303_route(route_line: RouteLine) -> tuple[ParsingResultType, Flag] | None:
    if Link.L303.value not in route_line.links:
        return None

    calc_properties = L303Properties(0)

    value = route_line.get(36, 0)
    if value == -1:
        return {}, calc_properties

    value, multipole_ints = value // 10, value % 10
    if multipole_ints >= 0:
        calc_properties |= L303Properties.calculates_multipole_ints

    return {}, calc_properties


l303_parser_funcs_general: dict[str, list[Callable]] = {
    'l303': [
        read_multipole_matrices,
    ],
}


l303_postprocess_general = {
    'l303': [
        postprocess_multipole_integrals,
    ]
}
