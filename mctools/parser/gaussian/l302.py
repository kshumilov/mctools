import os.path
import re

from typing import TextIO

import numpy as np

from ..lib import ParsingResult
from .utils import read_matrix_in_file

__all__ = [
    'l302_parser_funcs_general',
    'l302_parser_funcs_x2c',
    'l302_parser_funcs_all',
]

ovlp_header = re.compile(r'\*\*\* Overlap \*\*\*')
kinetic_header = re.compile(r'\*\*\* Kinetic Energy \*\*\*')
hcore_header = re.compile(r'\*\*\*\*\*\* Core Hamiltonian \*\*\*\*\*\*')
veffp_header = re.compile(r'\*\*\* Veff \(p space\) \*\*\*')
trelr_header = re.compile(r'\*\*\* Trel \(r space\) \*\*\*')
veffr_header = re.compile(r'\*\*\* Veff \(r space\) \*\*\*')
so_header = re.compile(r'\*\*\* SO unc. \*\*\*')  # Appears 3 times
x2c_header = re.compile(r'DK / X2C integrals')  # Appears 5 times
ortho_header = re.compile(r'Orthogonalized basis functions:')


def read_ao_overlap_matrix(file: TextIO, /,
                           restriction: str = 'R', *,
                           first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, ovlp_header, first_line=first_line)
    match restriction:
        case 'G':
             I2 = np.eye(2)
             matrix = np.kron(matrix, I2)  # (#AOs, #AOs)

    return dict(overlap=matrix), line


def read_ao_kinetic_energy_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, kinetic_header, first_line=first_line)
    return dict(kinetic=matrix), line


def read_ao_hcore_matrix(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, hcore_header, first_line=first_line)
    return dict(hcore=matrix), line


def read_veff_p_matrix(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, veffp_header, first_line=first_line)
    return dict(veff_p=matrix), line


def read_trel_r_matrix(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, trelr_header, first_line=first_line)
    return dict(trel_r=matrix), line


def read_veff_r_matrix(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, veffr_header, first_line=first_line)
    return dict(veff_r=matrix), line


def read_so_unc_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    result: ParsingResult = {}

    line = first_line
    for i in range(1, 4):
        matrix, line = read_matrix_in_file(file, so_header, first_line=line)
        result[f'so_unc_{i}'] = matrix

    return result, line


def read_x2c_matrices(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    result: ParsingResult = {}

    line = first_line
    for i in range(1, 6):
        matrix, line = read_matrix_in_file(file, x2c_header, first_line=line)
        result[f'x2c_{i}'] = matrix

    return result, line


def read_orthogonal_aos_matrix(file: TextIO, *, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, ortho_header, first_line=first_line)
    return dict(ortho=matrix), line


l302_parser_funcs_general = {
    'l302': [
        read_ao_overlap_matrix,
        read_ao_kinetic_energy_matrix,
        read_ao_hcore_matrix,
        read_orthogonal_aos_matrix,
    ]
}

l302_parser_funcs_x2c = {
    'l302': [
        read_veff_p_matrix,
        read_trel_r_matrix,
        read_veff_r_matrix,
        read_so_unc_matrices,
        read_ao_overlap_matrix,
    ]
}

l302_parser_funcs_all = {
    'l302': [
        read_ao_overlap_matrix,
        read_ao_kinetic_energy_matrix,
        read_ao_hcore_matrix,
        read_veff_p_matrix,
        read_trel_r_matrix,
        read_veff_r_matrix,
        read_so_unc_matrices,
        read_orthogonal_aos_matrix,
    ]
}
