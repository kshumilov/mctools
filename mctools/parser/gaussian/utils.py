from __future__ import annotations

import re

from itertools import chain
from typing import TextIO, Callable, TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ..lib import search_in_file, parse_file


if TYPE_CHECKING:
    from ..lib import ParsingResult, MatchDict

__all__ = [
    'process_complex_match',
    'read_matrix_in_file',

    'parse_link',
    'parse_gdvlog',
]


def process_complex_match(match_dict: MatchDict, *, real: str = 'real', imag: str = 'imag') -> complex:
    real = float(match_dict[real].replace('D', 'e'))
    imag = float(match_dict[imag].replace('D', 'e'))
    return real + imag * 1.j


mat_patt = re.compile(r'^[\s\d.D+\-]*$')
mat_idx_patt = re.compile(r'\s+(\d+)[\s\n]')


def read_matrix_in_file(file: TextIO, header: re.Pattern, first_line: str = '',
                        shape: tuple[int, ...] | None = None,
                        dtype: npt.DTypeLike = np.float_) -> tuple[np.ndarray, str]:
    # Finding matrix by header
    _, line = search_in_file(file, header, first_line=first_line,
                             err_msg=f"Couldn't find matrix header: {header.pattern}")

    cols_idx: list[int] = []
    raw_matrix: list[list[str]] = []

    while (line := file.readline()) and mat_patt.search(line):
        idx = list(map(int, mat_idx_patt.findall(line)))
        if len(idx) == 1 and '.' in line:
            values = line.split()[1:]

            if (row_idx := idx[0] - 1) < len(raw_matrix):
                raw_matrix[row_idx].extend(values)
            else:
                raw_matrix.append(values)
        else:
            cols_idx.extend([i - 1 for i in idx])

    n_rows, n_cols = len(raw_matrix), len(cols_idx)

    if n_rows == n_cols:
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


def parse_link(file: TextIO, link: str, read_funcs: list[Callable], result: ParsingResult, /,
               first_line: str = '') -> tuple[ParsingResult, str]:
    link_start_patt = re.compile(link_start_tmplt % link)
    _, line = search_in_file(file, link_start_patt, first_line=first_line,
                             err_msg=f'No link {link} information is found')
    return parse_file(file, read_funcs, result, first_line=line)


def parse_gdvlog(filename: str, links: dict[str, list[Callable]], /, **kwargs) -> ParsingResult:
    result: ParsingResult = {'source': filename, **kwargs}

    with open(filename, 'r') as file:
        print(f'Reading {filename}')

        line = ''
        for link, funcs in links.items():
            print(f'Parsing link {link}')
            data, line = parse_link(file, link, funcs, result, first_line=line)
            result.update(data)
    return result
