from collections import defaultdict
from typing import TextIO, Callable, Type

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..lib import (
    search_in_file,

    PatternNotFound,
    ProcessedPattern,

    simple_int_tmplt,
    simple_float_tmplt,

    ParsingResult,
    parse_file,
)

__all__ = [
    'fchk_parser_funcs_general',

    'read_main_info',
    'read_molecular_geometry',
    'read_ao_basis',
    'read_mo_coefficients',

    'parse_gdvfchk',

    'FCHK_ARRAY_PATT',
    'FCHK_SCALAR_PATT',
]


GAUSSIAN_SHELL_TYPES = {
    0: 'S',
    -1: 'SP', 1: 'P',
    -2: 'D',  2: 'd',
    -3: 'F',  3: 'f',
    -4: 'G',  4: 'g',
}

GAUSSIAN_ML_AO = {
    'S': [0],
    'P': [1, -1, 0],
    'D': [0, 1, -1, 2, -2],
    'F': [0, 1, -1, 2, -2, 3, -3],
    'G': [0, 1, -1, 2, -2, 3, -3, 4, -4]
}


def process_data_type(char: str, /) -> Type[np.float_ | np.int_ | np.str_]:
    match char:
        case 'R':
            return np.float_
        case 'I':
            return np.int_
        case 'C':
            return np.str_
        case _:
            raise ValueError('Invalid data_type')


FCHK_SCALAR_PATT = ProcessedPattern(
    r'%s\s*(?P<dt>I|R)\s*' + (r'(?P<v>%s|%s)' % (simple_int_tmplt, simple_float_tmplt)),
    constructor=lambda dt, v: dt(v), group_maps={'dt': process_data_type}
)

FCHK_ARRAY_PATT = ProcessedPattern(
    r'%s\s*(?P<data_type>I|R)\s*N=\s*(?P<N>\d+)',
    'fchk_array_info',
    group_maps={'N': int, 'data_type': process_data_type}
)


def read_fchk_scalar(file: TextIO, header: str, /, *, first_line: str = '') -> tuple[int|float|str, str]:
    patt = FCHK_SCALAR_PATT.update_pattern(header)
    return search_in_file(file, patt, first_line=first_line, err_msg=f'Could not find FCHK scalar: {header}')


def read_fchk_array(file: TextIO, header: str, /, *, first_line: str ='') -> tuple[np.ndarray, str]:
    patt = FCHK_ARRAY_PATT.update_pattern(header)
    fchk_arr_info, line = search_in_file(file, patt, first_line=first_line, err_msg=f'Could not find FCHK array: {header}')

    arr = []
    while len(arr) < fchk_arr_info.N:
        line = file.readline()
        arr.extend(line.split())

    arr = np.asarray(arr, dtype=fchk_arr_info.data_type)
    return arr, line


method_patt = ProcessedPattern(
    r'(?P<calc>[a-zA-Z]+)\s*(?P<method>(?P<restriction>R|RO|U|G)\w+)\s*(?P<basis>\w+)'
)


def read_fchk_header(file: TextIO, /) -> tuple[ParsingResult, str]:
    short_title = file.readline()

    line = file.readline()
    method = method_patt.search(line)
    if method is None:
        raise PatternNotFound('Could not find FCHK method line', line=line)
    return {'short_title': short_title, **method}, line


def read_main_info(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    short_title = file.readline()

    line = file.readline()
    method = method_patt.search(line)
    if method is None:
        raise PatternNotFound('Could not find FCHK method line', line=line)

    charge, line = read_fchk_scalar(file, 'Charge')
    multiplicity, line = read_fchk_scalar(file, 'Multiplicity', first_line=line)

    n_elec, line = read_fchk_scalar(file, 'Number of electrons', first_line=line)
    n_elec_a, line = read_fchk_scalar(file, 'Number of alpha electrons', first_line=line)
    n_elec_b, line = read_fchk_scalar(file, 'Number of beta electrons', first_line=line)

    n_ao, line = read_fchk_scalar(file, 'Number of basis functions', first_line=line)

    return {
        **header_info,
        'charge': charge, 'multiplicity': multiplicity,
        'n_elec': n_elec, 'n_elec_a': n_elec_a, 'n_elec_b': n_elec_b,
        'n_ao': n_ao  # TODO: select convention on AO representation: spherical or cartesian
    }, line


def read_molecular_geometry(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    atomic_number, line = read_fchk_array(file, 'Atomic numbers', first_line=first_line)
    coords, line = read_fchk_array(file, 'Current cartesian coordinates', first_line=line)
    coords = coords.reshape((len(atomic_number), 3))
    return {'atomic_number': atomic_number, 'coords': coords}, line


def read_ao_basis(file: TextIO, /,
                  atomic_number: npt.ArrayLike | None = None,
                  restriction: str = 'R', *,
                  first_line: str = '') -> tuple[ParsingResult, str]:
    # Shell information (Part 1)
    shell_code, line = read_fchk_array(file, 'Shell types', first_line=first_line)
    shell_size, line = read_fchk_array(file, 'Number of primitives per shell', first_line=line)
    shell2atom, line = read_fchk_array(file, 'Shell to atom map', first_line=line)
    shell2atom -= 1  # Start atom enumeration from 0

    # Primitive information (Part 2)
    primitive_exponents, line = read_fchk_array(file, 'Primitive exponents', first_line=line)
    contraction_coeffs, line = read_fchk_array(file, 'Contraction coefficients', first_line=line)

    shell_coords, line = read_fchk_array(file, 'Coordinates of each shell', first_line=line)
    shell_coords = shell_coords.reshape((len(shell_code), 3))

    shells = pd.DataFrame({
        'code': shell_code,
        'n_prim': shell_size,
        'atom': shell2atom,
    })
    shells['type'] = shells['code'].map(GAUSSIAN_SHELL_TYPES)
    shells[['x', 'y', 'z']] = shell_coords

    # Atomic Orbitals information
    aos = defaultdict(list)
    for shell_idx, shell in shells.iterrows():
        for shell_part in shell.type:
            mls = GAUSSIAN_ML_AO[shell_part.capitalize()]
            l = np.max(mls)
            n_ao_per_shell = len(mls)

            aos['shell'].extend([shell_idx] * n_ao_per_shell)
            aos['atom'].extend([shell.atom] * n_ao_per_shell)
            aos['l'].extend([l] * n_ao_per_shell)
            aos['ml'].extend(mls)

    aos = pd.DataFrame(aos)

    if atomic_number is not None:
        atomic_number = np.asarray(atomic_number)
        shells['element'] = atomic_number[shells['atom']]
        aos['element'] = atomic_number[aos['atom']]

    if restriction == 'G':
        aos = aos.loc[aos.index.repeat(2)].reset_index(drop=True)
        aos['spin'] = aos.index % 2

    return {
        'df_shells': shells,
        'df_ao': aos,
        'exponents': primitive_exponents,
        'coefficients': contraction_coeffs,
    }, line


def read_mo_coefficients(file: TextIO, n_ao: int, /,
                         restriction: str = 'R', *,
                         first_line: str = '') -> tuple[ParsingResult, str]:
    raw_mo, line = read_fchk_array(file, 'Alpha MO coefficients', first_line=first_line)

    match restriction:
        case 'R':
            C_MO = raw_mo.reshape(-1, n_ao)
        case 'U':
            raw_mo_b, line = read_fchk_array(file, 'Beta MO coefficients', first_line=line)
            C_MO = np.zeros((n_ao, n_ao), dtype=np.float_)
            C_MO[0::2, 0::2] = raw_mo.reshape(-1, n_ao)
            C_MO[1::2, 1::2] = raw_mo_b.reshape(-1, n_ao)
        case 'G':
            C_MO = raw_mo[0::2] + raw_mo[1::2] * 1.j
            C_MO = C_MO.reshape(-1, n_ao * 2)  # (#MOs, #AOs)
        case _:
            raise ValueError(f'Invalid MO restriction is specified: {restriction}')

    return {'C_MO': C_MO}, line


fchk_parser_funcs_general = [
    read_main_info,
    read_molecular_geometry,
    read_ao_basis,
    read_mo_coefficients,
]


def parse_gdvfchk(filename: str, read_funcs: list[Callable], /, **kwargs) -> ParsingResult:
    result: ParsingResult = {'source': filename, **kwargs}

    with open(filename, 'r') as file:
        print(f'Reading {filename}')
        result, line = parse_file(file, read_funcs, result)

    return result


if __name__ == '__main__':
    import os

    data_dir = os.path.join('..', '..', '..', 'data', '../../../example/fchk')
    gdvfchk = os.path.join(data_dir, 'casscf.fchk')

    result = parse_gdvfchk(gdvfchk, fchk_parser_funcs_general)
