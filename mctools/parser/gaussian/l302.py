import os.path
import re

from typing import TextIO

from .utils import read_matrix_in_file, ParsingResult

__all__ = [
    'l302_parser_funcs_general',
    'l302_parser_funcs_x2c',
    'l302_parser_funcs_all',
]

ovlp_header = re.compile(r'\*\*\* Overlap \*\*\*')


def read_overlap_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, ovlp_header, first_line=first_line)
    return dict(overlap=matrix), line

kinetic_header = re.compile(r'\*\*\* Kinetic Energy \*\*\*')


def read_kinetic_energy_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, kinetic_header, first_line=first_line)
    return dict(kinetic=matrix), line


hcore_header = re.compile(r'\*\*\*\*\*\* Core Hamiltonian \*\*\*\*\*\*')


def read_hcore_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, hcore_header, first_line=first_line)
    return dict(hcore=matrix), line


veffp_header = re.compile(r'\*\*\* Veff \(p space\) \*\*\*')


def read_veff_p_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, veffp_header, first_line=first_line)
    return dict(veff_p=matrix), line


trelr_header = re.compile(r'\*\*\* Trel \(r space\) \*\*\*')


def read_trel_r_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, trelr_header, first_line=first_line)
    return dict(trel_r=matrix), line


veffr_header = re.compile(r'\*\*\* Veff \(r space\) \*\*\*')


def read_veff_r_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, veffr_header, first_line=first_line)
    return dict(veff_r=matrix), line


so_header = re.compile(r'\*\*\* SO unc. \*\*\*')  # Appears 3 times


def read_so_unc_matrices(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    result: ParsingResult = {}

    line = first_line
    for i in range(1, 4):
        matrix, line = read_matrix_in_file(file, so_header, first_line=line)
        result[f'so_unc_{i}'] = matrix

    return result, line


x2c_header = re.compile(r'DK / X2C integrals')  # Appears 5 times


def read_x2c_matrices(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    result: ParsingResult = {}

    line = first_line
    for i in range(1, 6):
        matrix, line = read_matrix_in_file(file, x2c_header, first_line=line)
        result[f'x2c_{i}'] = matrix

    return result, line


ortho_header = re.compile(r'Orthogonalized basis functions:')


def read_orthogonal_aos_matrix(file: TextIO, first_line: str = '') -> tuple[ParsingResult, str]:
    matrix, line = read_matrix_in_file(file, ortho_header, first_line=first_line)
    return dict(ortho=matrix), line


l302_parser_funcs_general = {
    'l302': [
        read_overlap_matrix,
        read_kinetic_energy_matrix,
        read_hcore_matrix,
        read_orthogonal_aos_matrix,
    ]
}

l302_parser_funcs_x2c = {
    'l302': [
        read_veff_p_matrix,
        read_trel_r_matrix,
        read_veff_r_matrix,
        read_so_unc_matrices,
        read_overlap_matrix,
    ]
}

l302_parser_funcs_all = {
    'l302': [
        read_overlap_matrix,
        read_kinetic_energy_matrix,
        read_hcore_matrix,
        read_veff_p_matrix,
        read_trel_r_matrix,
        read_veff_r_matrix,
        read_so_unc_matrices,
        read_orthogonal_aos_matrix,
    ]
}


if __name__ == '__main__':
    data_dir = os.path.join('data')
    gdvlog = os.path.join(data_dir, 'casscf.log')

    result: dict[ParsingResult] = {}
    with open(gdvlog, 'r') as f:
        overlap_result, line = read_overlap_matrix(f)
        result.update(overlap_result)

        kinetic_result, line = read_kinetic_energy_matrix(f, first_line=line)
        result.update(kinetic_result)

        hcore_result, line = read_hcore_matrix(f, first_line=line)
        result.update(hcore_result)

    for label, matrix in result.items():
        print(label)
        print(matrix)
