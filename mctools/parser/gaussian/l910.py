import math
import re

from functools import reduce
from typing import TextIO, Callable

import numpy as np
import pandas as pd
from scipy import sparse

from mctools.core import MCStates, MCPeaks

from ..lib import (
    find_line_in_file,
    find_pattern_in_file,

    ParsingResult,

    grouped_tmplt,
    simple_float_tmplt,
    simple_int_tmplt,
    simple_bool_tmplt,
)

from .utils import float_patt, bool_map, process_complex_match, process_float_match, read_matrix_in_file

__all__ = [
    'l910_parser_funcs_general',
    'l910_parser_funcs_rdms',

    'read_rdms',
    'read_mc_spec',
    'read_ci',
    'read_rdm_diags',
    'read_oscillator_strength',
]

mc_spec_start_patt = re.compile(r'Input Summary:')

orb_type_patt = re.compile(r'\s*'.join([
    r'RHF=\s*%s\s*,' % (grouped_tmplt % (r'is_rhf', simple_bool_tmplt)),
    r'CRHF=\s*%s\s*,' % (grouped_tmplt % (r'is_crhf', simple_bool_tmplt)),
    r'ROHF=\s*%s\s*,' % (grouped_tmplt % (r'is_rohf', simple_bool_tmplt)),
    r'GHF=\s*%s\s*' % (grouped_tmplt % (r'is_ghf', simple_bool_tmplt)),
]))

ci_type_patt = re.compile(r''.join([
    r'CAS=\s*%s\s*,\s*' % (grouped_tmplt % (r'is_cas', simple_bool_tmplt)),
    r'RAS=\s*%s\s*,\s*' % (grouped_tmplt % (r'is_ras', simple_bool_tmplt)),
    r'MRCISD=\s*%s\s*' % (grouped_tmplt % (r'is_mrcisd', simple_bool_tmplt)),
]))

ras_patt = re.compile(r'RAS\(\s*%s,\s*%s,\s*%s\)' % (
    grouped_tmplt % (r'n_ras1', simple_int_tmplt),
    grouped_tmplt % (r'n_ras2', simple_int_tmplt),
    grouped_tmplt % (r'n_ras3', simple_int_tmplt),
))

cas_patt = re.compile(r'CAS\(\s*%se\s*,\s*%so\s*\)' % (
    grouped_tmplt % (r'n_e', simple_int_tmplt),
    grouped_tmplt % (r'n_o', simple_int_tmplt),
))

orb_info_patt = re.compile(r''.join([
    r'NTOrb=\s*%s\s*' % grouped_tmplt % (r'n_orb', simple_int_tmplt),
    r'NIOrb=\s*%s\s*' % grouped_tmplt % (r'n_inact', simple_int_tmplt),
    r'NAOrb=\s*%s\s*' % grouped_tmplt % (r'n_act', simple_int_tmplt),
    r'NVOrb=\s*%s\s*' % grouped_tmplt % (r'n_virt', simple_int_tmplt),
]))

sa_start_patt = re.compile(r'SA Weights Read:')
sa_weight_patt = re.compile(r''.join([
    'State:\s*%s\s*' % grouped_tmplt % ('state_idx', simple_int_tmplt),
    'Weight:\s*%s\s*' % (float_patt % 'weight'),
]))

tot_elec_info_patt = re.compile(r''.join([
    r'Electrons, Alpha=\s*%s\s*Beta=\s*%s' % (
        grouped_tmplt % (r'n_elec_a', simple_int_tmplt),
        grouped_tmplt % (r'n_elec_b', simple_int_tmplt),
    )
]))

act_elec_info_patt = re.compile(r''.join([
    r'Active Electrons, Alpha=\s*%s\s*Beta=\s*%s' % (
        grouped_tmplt % (r'n_elec_a', simple_int_tmplt),
        grouped_tmplt % (r'n_elec_b', simple_int_tmplt),
    )
]))

n_confgis_patt = re.compile(r'Alpha Strings=\s*%s\s*Beta Strings=\s*%s\s*' % (
    grouped_tmplt % (r'n_config_a', simple_int_tmplt),
    grouped_tmplt % (r'n_config_b', simple_int_tmplt),
))


def read_mc_spec(file: TextIO, /, first_line: str = '') -> tuple[ParsingResult, str]:
    match, line = find_line_in_file(file, mc_spec_start_patt, first_line=first_line)
    if match is None:
        raise ValueError('No MCSCF Specification is found')

    mc_spec = {}

    orb_type_info, line = find_pattern_in_file(file, orb_type_patt, first_line=line, default_group_map=bool_map)
    ci_type_info, line = find_pattern_in_file(file, ci_type_patt, first_line=line, default_group_map=bool_map)

    if ci_type_info['is_cas']:
        mc_patt = cas_patt
    else:
        mc_patt = ras_patt

    mc_info, line = find_pattern_in_file(file, mc_patt, first_line=line, default_group_map=int)
    if ci_type_info['is_cas']:
        ras_mo = (0, mc_info['n_o'], 0)
    else:
        ras_mo = (mc_info['n_ras1'], mc_info['n_ras2'], mc_info['n_ras3'])

    mc_spec.update({'ras': ras_mo, 'n_mos': sum(ras_mo)})

    orb_info, line = find_line_in_file(file, orb_info_patt, first_line=line, default_group_map=int)
    mo = (orb_info['n_inact'], orb_info['n_act'], orb_info['n_virt'])
    mc_spec['mo_all'] = mo

    match, line = find_line_in_file(file, sa_start_patt, first_line=line, n_skips=1)
    if match is not None:
        sa_weights_info, line = find_pattern_in_file(file, sa_weight_patt,
                                                     group_maps={'weight': float,},
                                                     until_first_failure=True, max_matches=math.inf)
        idx = []
        sa_weights = []
        while sa_weights_info:
            match = sa_weights_info.pop()
            idx.append(match.pop('state_idx'))
            sa_weights.append(match.pop('weight'))

        idx = np.argsort(idx)
        sa_weights = np.asarray(sa_weights)[idx]

        mc_spec['sa_weights'] = sa_weights

    tot_elec_info, line = find_pattern_in_file(file, tot_elec_info_patt, first_line=line, default_group_map=int)
    n_elec = (tot_elec_info['n_elec_a'], tot_elec_info['n_elec_b'])

    act_elec_info, line = find_pattern_in_file(file, act_elec_info_patt, first_line=line, default_group_map=int)
    n_elec_act = (act_elec_info['n_elec_a'], act_elec_info['n_elec_b'])

    n_configs_info, line = find_pattern_in_file(file, n_confgis_patt, first_line=line, default_group_map=int)
    n_config = (n_configs_info['n_config_a'], n_configs_info['n_config_b'])

    mc_spec.update({
        'elec': n_elec_act, 'n_elec': sum(n_elec_act), 'elec_all': n_elec,
        'config': n_config, 'n_configs': reduce(lambda x, y: x * y, n_config)
    })
    return mc_spec, line


mc_done_patt = re.compile(r'[CR]AS(SCF|CI) Done:')
state_patt = re.compile(
    r'State:\s*(?P<idx>\d*)\s*Energy \(Hartree\):\s*%s' % (grouped_tmplt % ('E', simple_float_tmplt))
)
ci_complex_patt = r'\s*'.join((
    grouped_tmplt % (r'real', simple_float_tmplt),
    grouped_tmplt % (r'imag', simple_float_tmplt),
))
ci_vec_patt = re.compile(r'\(\s*(?P<addr>\d*)\)\s*%s' % ci_complex_patt)


def read_ci(file: TextIO, n_configs: int, /, max_det: int = 50, first_line: str = '') -> tuple[ParsingResult, str]:
    match, line = find_line_in_file(file, mc_done_patt, first_line=first_line)
    if match is None or not line:
        raise ValueError('No state information is found')

    # State energy info
    energy: list[float] = []
    state_idx: list[int] = []

    # CI Vectors data in COO format
    ci_data: list[complex | float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    while True:
        energy_info, line = find_pattern_in_file(file, state_patt, group_maps={'idx': int, 'E': float}, n_skips=2,
                                                 max_matches=1, until_first_failure=bool(state_idx), first_line=line)
        if not energy_info:
            break

        energy.append(energy_info['E'])
        state_idx.append(energy_info['idx'])

        ci_vec_info, line = find_pattern_in_file(file, ci_vec_patt,
                                                 group_maps={'addr': int}, match_funcs={'C': process_complex_match},
                                                 max_matches=max_det, until_first_failure=True)

        for state_info in ci_vec_info:
            ci_data.append(state_info['C'])
            row_idx.append(energy_info['idx'] - 1)
            col_idx.append(state_info['addr'] - 1)

    n_states = max(state_idx)
    ci_vecs = sparse.coo_array((ci_data, (row_idx, col_idx)), shape=(n_states, n_configs))

    df = pd.DataFrame({MCStates.E_COL: energy, MCStates.STATE_COL: state_idx})
    return dict(df_states=df, ci_vecs=ci_vecs, n_states=n_states), line


rdm_start_patt = re.compile(r'\*\* Printing Density Matrices for all States \*\*')
rdm_diag_patt = re.compile(r'For Simplicity: The diagonals of 1PDM for State:\s*(?P<idx>\d*)')


def read_rdm_diags(file: TextIO, n_states: int, n_mos: int, /, first_line: str = '') -> tuple[ParsingResult, str]:
    match, line = find_line_in_file(file, rdm_start_patt, first_line=first_line)
    if match is None:
        raise ValueError('No PDM information is found')

    rdm_diags: np.ndarray = np.empty((n_states, n_mos), dtype=np.float_)
    for i in range(n_states):
        rdm_diag, line = read_matrix_in_file(file, rdm_diag_patt, shape=(n_mos,))
        rdm_diags[i] = rdm_diag

    return dict(rdm_diags=rdm_diags), line


rdm_real_patt = re.compile('1PDM Matrix \(real\) :')
rdm_imag_patt = re.compile('1PDM Matrix \(imag\) :')


def read_rdms(file: TextIO, n_states: int, n_mos: int, /, first_line: str = '') -> tuple[ParsingResult, str]:
    match, line = find_line_in_file(file, rdm_start_patt, first_line=first_line)
    if match is None:
        raise ValueError('No PDM information is found')

    rdms: np.ndarray = np.zeros((n_states, n_mos, n_mos), dtype=np.complex128)
    for i in range(n_states):
        rdm_real, line = read_matrix_in_file(file, rdm_real_patt, shape=(n_mos, n_mos), first_line=line)
        rdm_imag, line = read_matrix_in_file(file, rdm_imag_patt, shape=(n_mos, n_mos), first_line=line)
        rdms[i] = rdm_real + rdm_imag * 1.j

    return {
        'rdms': rdms,
        'rdm_diags': np.diagonal(rdms, axis1=1, axis2=2)
    }, line


osc_patt_start = re.compile(r'Using Dipole Ints in file:')
osc_patt = re.compile(
    r'Oscillator Strength For States\s*(?P<initial>\d*)\s*:\s*(?P<final>\d*)\s*f=\s*(?P<osc>[+\-]?\d*\.\d*)'
)


def read_oscillator_strength(file: TextIO, n_states: int, n_ground: int, /,
                             first_line: str = '') -> tuple[ParsingResult, str]:
    match, line = find_line_in_file(file, osc_patt_start, first_line=first_line)
    if match is None:
        raise ValueError('No Oscillator information is found')

    osc_info, line = find_pattern_in_file(file, osc_patt,
                                          first_line=line, max_matches=n_states * n_ground,
                                          group_maps={'osc': float}, default_group_map=int)

    initial_state: list[int] = []
    final_state: list[int] = []
    osc_strength: list[float] = []

    for d in osc_info:
        initial_state.append(d['initial'])
        final_state.append(d['final'])
        osc_strength.append(d['osc'])

    df = pd.DataFrame({
        MCPeaks.INITIAL_STATE_COL: initial_state,
        MCPeaks.FINAL_STATE_COL: final_state,
        MCPeaks.OSC_COL: osc_strength
    })
    return dict(df_peaks=df), line


spin_header_patt = re.compile(r'Computing Spin expectation values.')
spin_patt = re.compile(
    r'State:\s*(?P<state>\d*)\s*'
    r'<Sx>=\s*(?P<sx>[+\-]?\d*\.\d*)\s*'
    r'<Sy>=\s*(?P<sy>[+\-]?\d*\.\d*)\s*'
    r'<Sz>=\s*(?P<sz>[+\-]?\d*\.\d*)\s*'
    r'<Sx\*\*2>=\s*(?P<sx_sq>[+\-]?\d*\.\d*)\s*'
    r'<Sy\*\*2>=\s*(?P<sy_sq>[+\-]?\d*\.\d*)\s*'
    r'<Sz\*\*2>=\s*(?P<sz_sq>[+\-]?\d*\.\d*)\s*'
    r'<S\*\*2>=\s*(?P<s_sq>[+\-]?\d*\.\d*)\s*'
    r'S=\s*(?P<s>[+\-]?\d*\.\d*)'
)

# TODO: include Spin Parsing


l910_parser_funcs_general: dict[str, list[Callable]] = {
    'l910': [
        read_mc_spec,
        read_ci,
        read_rdm_diags,
        read_oscillator_strength
    ],
}

l910_parser_funcs_rdms: dict[str, list[Callable]] = {
    'l910': [
        read_mc_spec,
        read_ci,
        read_rdms,
        read_oscillator_strength,
    ],
}

if __name__ == '__main__':
    import os

    data_dir = os.path.join('..', '..', '..', 'data')
    gdvlog = os.path.join(data_dir, 'example.log')

    with open(gdvlog, 'r') as file:
        result, line = read_mc_spec(file, gdvlog)

        print(result)
