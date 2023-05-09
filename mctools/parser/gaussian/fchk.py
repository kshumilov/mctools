import re

import numpy as np
import pandas as pd

from typing import TextIO, Callable

from ..lib import (
    find_pattern_in_file,
    find_pattern_in_line,

    simple_int_tmplt,
    simple_float_tmplt,

    ParsingResult,
    parse_file,
)

__all__ = [
    'fchk_parser_funcs_general',

    'read_main_info',
    'read_geom',
    'read_ao_basis',
    'read_mo_coefficients',

    'parse_gdvfchk',
]


def read_fchk_scalar(file: TextIO, header: str, /, first_line: str = '') -> tuple[int|float|str, str]:
    patt = re.compile(r'%s\s*(?P<data_type>I|R)\s*(?P<value>%s|%s)' %
                      (header, simple_int_tmplt, simple_float_tmplt))
    match, line = find_pattern_in_file(file, patt, first_line=first_line, group_maps={'N': int})
    if match is None:
        raise ValueError(f"Couldn't find header: {header}")

    match match.pop('data_type'):
        case 'R':
            data_type = float
        case 'I':
            data_type = int
        case 'C':
            data_type = str
        case _:
            raise ValueError('Invalid data_type')

    value = data_type(match.pop('value'))
    return value, line


def read_fchk_array(file: TextIO, header: str, /, first_line='') -> tuple[np.ndarray, str]:
    patt = re.compile(r'%s\s*(?P<data_type>I|R)\s*N=\s*(?P<N>\d+)' % header)
    match, line = find_pattern_in_file(file, patt, first_line=first_line, group_maps={'N': int})
    if match is None:
        raise ValueError(f"Couldn't find header: {header}")

    arr = []
    N = match.pop('N')
    while len(arr) < N:
        line = file.readline()
        arr.extend(line.split())

    match match.pop('data_type'):
        case 'R':
            data_type = np.float64
        case 'I':
            data_type = np.int64
        case _:
            raise ValueError('Invalid data_type')

    arr = np.asarray(arr, dtype=data_type)
    return arr, line


method_patt = re.compile(r'(?P<calc>[a-zA-Z]+)\s*(?P<method>(?P<restriction>R|RO|U|G)\w+)\s*(?P<basis>\w+)')


def read_main_info(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    short_title = file.readline()
    method, *_ = find_pattern_in_line(file.readline(), method_patt)
    charge, line = read_fchk_scalar(file, 'Charge')
    multiplicity, line = read_fchk_scalar(file, 'Multiplicity', first_line=line)

    n_elec, line = read_fchk_scalar(file, 'Number of electrons', first_line=line)
    n_elec_a, line = read_fchk_scalar(file, 'Number of alpha electrons', first_line=line)
    n_elec_b, line = read_fchk_scalar(file, 'Number of beta electrons', first_line=line)

    n_ao, line = read_fchk_scalar(file, 'Number of basis functions', first_line=line)

    return {
        'short_title': short_title,
        **method,
        'charge': charge, 'multiplicity': multiplicity,
        'n_elec': n_elec, 'n_elec_a': n_elec_a, 'n_elec_b': n_elec_b,
        'n_ao': n_ao  # TODO: select convention on AO representation: spherical or cartesian
    }, line


def read_geom(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    Z, line = read_fchk_array(file, 'Atomic numbers', first_line=first_line)
    coords, line = read_fchk_array(file, 'Current cartesian coordinates', first_line=line)
    coords = coords.reshape((len(Z), 3))
    return {'Z': Z, 'coords': coords}, line


def read_ao_basis(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    # Shell information
    shell_types, line = read_fchk_array(file, 'Shell types', first_line=first_line)
    shell_size, line = read_fchk_array(file, 'Number of primitives per shell', first_line=line)
    shell2atom, line = read_fchk_array(file, 'Shell to atom map', first_line=line)
    shell2atom -= 1  # Start atom enumeration from 0

    df_shells = pd.DataFrame({
        'l': shell_types,  # FIXME: doesn't handle cartesian AOs
        'n_prim': shell_size,
        'atom': shell2atom,
    })

    # Primitive information
    # primitive_exponents, line = read_fchk_array(file, 'Primitive exponents', first_line=line)
    # contraction_coeffs, line = read_fchk_array(file, 'Contraction coefficients', first_line=line)
    # shell_coords, line = read_fchk_array(file, 'Coordinates of each shell', first_line=line)
    # shell_coords = shell_coords.reshape((len(shell_types), 3))
    #
    # df_primitives = pd.DataFrame({
    #     'exp': primitive_exponents,
    #     'coeff': contraction_coeffs,
    # })
    # df_primitives[['x', 'y', 'z']] = shell_coords

    # # Magnetic QN
    # # FIXME: px != p-1
    # max_l = shell2l.max()
    # mls = np.zeros(2 * max_l + 1, dtype=np.int64)
    # mls[1::2] = np.arange(1, max_l + 1)
    # mls[2::2] = mls[1::2]
    #
    # aos_per_shell = 2 * shell2l + 1
    # n_spin = 1  # Kramer's Unrestricted formalism
    #
    # n_ao = np.sum(aos_per_shell)
    # n_ao *= n_spin  # TODO: select convention on AO representation
    # ao_idx = np.arange(n_ao)
    #
    # # AO Mappings
    # ao2spin = ao_idx % n_spin
    # ao2l = np.zeros((n_ao,), dtype=np.int64)
    # ao2ml = np.zeros((n_ao,), dtype=np.int64)
    # ao2atom = np.zeros((n_ao,), dtype=np.int64)
    #
    # offset = 0
    # for shell_idx in range(len(shell_types)):
    #     n = aos_per_shell[shell_idx]
    #     end = offset + n_spin * n
    #
    #     atom = shell2atom[shell_idx]
    #     l = shell2l[shell_idx]
    #     ml = mls[:2 * l + 1]
    #
    #     # Iterate over spins and fill arrays
    #     for i in range(n_spin):
    #         sl = np.s_[offset + i:end + i:n_spin]
    #         ao2atom[sl] = atom  # Alpha
    #         ao2l[sl] = l
    #         ao2ml[sl] = ml
    #
    #     offset = end
    #
    # df_ao = pd.DataFrame({
    #     'atom': ao2atom,
    #     'l': ao2l,
    #     'ml': ao2ml,
    #     's': ao2spin,
    # }, index=ao_idx)

    return {'df_shells': df_shells}, line


def read_mo_coefficients(file: TextIO, n_ao: int, restriction: str, /, first_line: str = '') -> tuple[ParsingResult, str]:
    raw_mo, line = read_fchk_array(file, 'Alpha MO coefficients', first_line=first_line)

    match restriction:
        case 'R':
            C_MO = raw_mo.reshape(n_ao, n_ao)
            # C_MO = np.kron(C_MO, np.eye(2, dtype=np.float_))
        case 'U':
            raw_mo_b, line = read_fchk_array(file, 'Beta MO coefficients', first_line=line)
            C_MO = np.zeros((n_ao, n_ao), dtype=np.float_)
            C_MO[0::2, 0::2] = raw_mo.reshape(n_ao, n_ao)
            C_MO[1::2, 1::2] = raw_mo_b.reshape(n_ao, n_ao)
        case 'G':
            C_MO = raw_mo[0::2] + raw_mo[1::2] * 1.j
            C_MO = C_MO.reshape(n_ao * 2, n_ao * 2)  # (#MOs, #AOs)
        case _:
            raise ValueError(f'Invalid MO restriction is specified: {restriction}')

    return {'C_MO': C_MO}, line


fchk_parser_funcs_general = [
    read_main_info,
    read_geom,
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

    data_dir = os.path.join('..', '..', '..', 'data', 'fchk')
    gdvfchk = os.path.join(data_dir, 'casscf.fchk')

    result = parse_gdvfchk(gdvfchk, fchk_parser_funcs_general)
