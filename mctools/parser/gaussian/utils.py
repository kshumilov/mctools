import inspect
import re
import warnings
from itertools import chain
from typing import TextIO, Callable

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ..lib import MatchDict, find_line_in_file, simple_float_tmplt, simple_int_tmplt

__all__ = [
    'process_float_match',
    'process_complex_match',
    'read_matrix_in_file',

    'bool_map',

    'ParsingResult',

    'parse_link',
    'parse_gdvlog',
]

bool_map = lambda x: x.capitalize() == 'T'

float_patt = r'(?P<%s>%s(D%s)?)' % (r'%s', simple_float_tmplt, simple_int_tmplt)

ParsingResult = dict[str, np.ndarray | sparse.coo_array]


def process_float_match(match_dict: MatchDict, key: str = 'float_value') -> float:
    if key not in match_dict:
        raise ValueError(f'No {key} present in the match_dict.')
    return float(match_dict.pop(key).replace('D', 'e'))


def process_complex_match(match_dict: MatchDict) -> complex:
    real = process_float_match(match_dict, key='real')
    imag = process_float_match(match_dict, key='imag')
    return real + imag * 1.j


mat_patt = re.compile(r'^[\s\d.D+\-]*$')
mat_idx_patt = re.compile(r'\s+(\d+)[\s\n]')


def read_matrix_in_file(file: TextIO, first_line: str = '', is_square: bool = True,
                        shape: tuple[int, ...] | None = None,
                        dtype: npt.DTypeLike = np.float_) -> tuple[np.ndarray, str]:
    # TODO: implement finding matrix by header

    cols_idx: list[int] = []
    raw_matrix: list[list[str]] = []

    while (line := (first_line if first_line else file.readline())) and mat_patt.search(line):
        idx = list(map(int, mat_idx_patt.findall(line)))
        if len(idx) == 1 and '.' in line:
            values = line.split()[1:]

            if (row_idx := idx[0] - 1) < len(raw_matrix):
                raw_matrix[row_idx].extend(values)
            else:
                raw_matrix.append(values)
        else:
            cols_idx.extend([i - 1 for i in idx])

        first_line = ''

    n_rows, n_cols = len(raw_matrix), len(cols_idx)

    if is_square:
        if n_cols > len(raw_matrix[0]):
            matrix = np.empty((n_rows, n_cols), dtype='<U15')

            # Lower Triangular Matrix
            il = np.tril_indices_from(matrix)
            matrix[il] = list(chain.from_iterable(raw_matrix))

            iu = np.triu_indices_from(matrix, k=1)
            matrix[iu] = matrix.T[iu]
        elif n_cols < len(raw_matrix[-1]):
            matrix = np.empty((n_rows, n_cols), dtype='<U15')

            # Upper Triangular Matrix
            iu = np.triu_indices_from(matrix)
            matrix[iu] = list(chain.from_iterable(raw_matrix))

            il = np.tril_indices_from(matrix, k=-1)
            matrix[il] = matrix.T[il]
        else:
            # Full Matrix
            matrix = np.asarray(raw_matrix, dtype='<U15')
    else:
        matrix = np.asarray(raw_matrix, dtype='<U15')

    matrix = np.char.replace(matrix, 'D', 'e').astype(dtype)
    if shape is not None:
        matrix = matrix.reshape(shape)

    return matrix, line


link_start_tmplt = r'\(Enter\s*[/\w\-]*%s\.exe\)'


def parse_link(file: TextIO, link: str, funcs: list[Callable], result: ParsingResult, /,
               first_line: str = '') -> tuple[ParsingResult, str]:
    link_start_patt = re.compile(link_start_tmplt % link)

    match, line = find_line_in_file(file, link_start_patt, first_line=first_line)
    if match is None:
        raise ValueError(f'No link {link} information is found')

    for func in funcs:
        args = []
        for param, info in inspect.signature(func).parameters.items():
            if param in {'file', 'first_line'}:
                continue
            elif info.kind == info.POSITIONAL_OR_KEYWORD or info.kind == info.KEYWORD_ONLY:
                continue

            # TODO: generalize to read kwargs too
            if param in result:
                args.append(result[param])
            else:
                print(f'Skipping: {func.__name__}, parameter {param} is unavailable')
                break
        else:
            try:
                print(f'Executing: {func.__name__}')
                data, line = func(file, *args, first_line=line)
                result.update(data)
            except ValueError as err:
                warnings.warn(err.args[0])

    return result, line


def parse_gdvlog(filename: str, links: dict[str, list[Callable]], /, **kwargs) -> ParsingResult:
    result: ParsingResult = {'source': filename, **kwargs}

    with open(filename, 'r') as file:
        print(f'Reading {filename}')

        line = ''
        for link, funcs in links.items():
            print(f'Parsing link {link}')
            try:
                data, line = parse_link(file, link, funcs, result, first_line=line)
                result.update(data)
            except ValueError as err:
                warnings.warn(err.args[0])

    return result


if __name__ == '__main__':
    import os

    data_dir = os.path.join('../..', '..', 'data')
    gdvlog = os.path.join(data_dir, '../../../data/example.log')

    # # Link 302
    # ovlp_header = re.compile(r'\*\*\* Overlap \*\*\*')
    # hcore_header = re.compile(r'\*\*\*\*\*\* Core Hamiltonian \*\*\*\*\*\*')
    # veffp_header = re.compile(r'\*\*\* Veff \(p space\) \*\*\*')
    # trelr_header = re.compile(r'\*\*\* Trel \(r space\) \*\*\*')
    # veffr_header = re.compile(r'\*\*\* Veff \(r space\) \*\*\*')
    # so_header = re.compile(r'\*\*\* SO unc. \*\*\*')  # Appears 3 times
    # x2c_header = re.compile(r'DK / X2C integrals')  # Appears 5 times
    # ortho_header = re.compile(r'Orthogonalized basis functions:')
    #
    # # Link 303
    # multipole_header = re.compile(r'Multipole matrices ')  # Appears 3 times
    # fermi_header = re.compile(r'Fermi contact integrals:')  # Not square
    #
    # # Read One Electron Integrals
    # if False:
    #     gdvlog = os.path.join('tests', 'data', 'example.log')
    #     with open(gdvlog, 'r') as file:
    #         # Link 302 Integrals
    #         match, line = find_line_in_file(file, ovlp_header)
    #         print(line)
    #         S, line = read_matrix_in_file(file)
    #         print(line)
    #         T, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         match, line = find_line_in_file(file, hcore_header)
    #         print(line)
    #         H, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         match, line = find_line_in_file(file, veffp_header)
    #         print(line)
    #
    #         Veff_p, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         Trel, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         Veff_r, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         SO1, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         SO2, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         SO3, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         x2c1, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         x2c2, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         x2c3, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         x2c4, line = read_matrix_in_file(file)
    #         print(line)
    #
    #         x2c5, line = read_matrix_in_file(file)
    #         print(line)

    # n_states, n_det = 2796, 2796
    # data_dir = os.path.join('..', '..',)
    # ci_coeff_dat = os.path.join(data_dir, 'tests', 'data', 'rasci_1.ci_coeff.dat')
    # ci_vecs_coo = read_ci_vectors(ci_coeff_dat, n_states, n_det, norm_error=-1.0)
    # ci_vecs = ci_vecs_coo.tocsr()
    #
    # matrix = []
    # for chunk in read_rwfdump(ci_coeff_dat, to_numpy=False):
    #     matrix.extend(chunk)

    # n_states, n_configs, n_mos = 2796, 2796, 36
    # df, pdm_diags, ci_vecs = read_l910_gdvlog(gdvlog, n_states, n_configs, n_mos)
