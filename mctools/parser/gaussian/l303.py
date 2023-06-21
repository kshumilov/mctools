import re

from itertools import permutations
from collections import defaultdict
from typing import TextIO, Callable

import numpy as np
import numpy.typing as npt

from ..lib import search_in_file, ParsingResult, ProcessedPattern, DIMS_MAP
from .utils import read_matrix_in_file

__all__ = [
    'MULTIPOLE_INTEGRALS',

    'read_multipole_matrices',
    'read_rxdel_matrices',

    'l303_parser_funcs_general',
    'l303_postprocess_general',
]


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


def read_multipole_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
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


rxp_header = re.compile(r'Calculate R x Del and Del integrals.')
rxp_int_header = re.compile(r'R x Del for IMat=\s*\d+:')


def read_rxdel_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, rxp_header, first_line=first_line,
                             err_msg='No r x del information is found')

    matrices = []
    for i in range(3):
        matrix, line = read_matrix_in_file(file, rxp_int_header, first_line=line)
        matrices.append(matrix)

    return {'rxdel': np.stack(matrices)}, line


def postprocess_multipole_integrals(result: ParsingResult, integrals_key: str = 'integrals') -> None:
    integrals = result.setdefault(integrals_key, {})
    for integral in MULTIPOLE_INTEGRALS + ['rxdel']:
        if integral in result:
            integrals[integral] = result.pop(integral)


fermi_header = re.compile(r'Fermi contact integrals:')  # Not square


l303_parser_funcs_general: dict[str, list[Callable]] = {
    'l303': [
        read_multipole_matrices,
        read_rxdel_matrices,
    ],
}


l303_postprocess_general = {
    'l303': [
        postprocess_multipole_integrals,
    ]
}
