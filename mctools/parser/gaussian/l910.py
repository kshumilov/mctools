from __future__ import annotations

import re, math

from functools import reduce
from typing import TextIO, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import sparse

from ...core.cistring import RASMOs
from ...core.mcpeaks import MCPeaks
from ...core.mcstates import MCStates
from ...core.mcspace import MOSpaces

from ..lib import (
    search_in_file,
    findall_in_file,
    ProcessedPattern,

    int_patt, simple_int_tmplt,
    float_patt,
    bool_patt,

    bool_map,
)

from .utils import (
    process_complex_match,
    read_matrix_in_file
)

if TYPE_CHECKING:
    from ..lib import ParsingResult


__all__ = [
    'l910_parser_funcs_general',
    'l910_parser_funcs_rdms',

    'read_rdms',
    'read_mc_spec',
    'read_ci_vecs',
    'read_rdm_diags',
    'read_oscillator_strength',
]

mc_spec_start_patt = re.compile(r'Input Summary:')

ref_type_patt = ProcessedPattern(r'\s*,\s*'.join([
    r'RHF=\s*%s' % (bool_patt % 'is_rhf'),
    r'CRHF=\s*%s' % (bool_patt % 'is_crhf'),
    r'ROHF=\s*%s' % (bool_patt % 'is_rohf'),
    r'GHF=\s*%s' % (bool_patt % 'is_ghf')
]), 'ref_type', default_group_map=bool_map)

ci_type_patt = ProcessedPattern(r'\s*'.join([
    r'CAS=\s*%s\s*,' % (bool_patt % r'is_cas'),
    r'RAS=\s*%s\s*,' % (bool_patt % r'is_ras'),
    r'MRCISD=\s*%s\s*' % (bool_patt % r'is_mrcisd'),
]), 'ci_type', default_group_map=bool_map)

ras_patt = ProcessedPattern(r'RAS\(\s*%s,\s*%s,\s*%s\)' % (
    (int_patt % r'r1'), (int_patt % r'r2'), (int_patt % r'r3'),
), RASMOs, default_group_map=int)

cas_patt = ProcessedPattern(r'CAS\(\s*%se\s*,\s*%so\s*\)' % (
    (int_patt % r'n_e'), (int_patt % r'n_o'),
), lambda n_e, n_o: RASMOs(0, n_o, 0), default_group_map=int)

ci_spaces_patt = ProcessedPattern(r'\s*'.join([
    r'NTOrb=\s*%s' % (int_patt % r'n'),
    r'NIOrb=\s*%s' % (int_patt % r'i'),
    r'NAOrb=\s*%s' % (int_patt % r'a'),
    r'NVOrb=\s*%s' % (int_patt % r'v'),
]) + r'\s*', lambda n, i, a, v: MOSpaces.from_spaces(a, i, v), default_group_map=int)

sa_start_patt = re.compile(r'SA Weights Read:')
sa_weight_patt = ProcessedPattern(r'\s*'.join([
    r'State:\s*%s' % simple_int_tmplt,
    r'Weight:\s*%s' % (float_patt % r'w'),
]) + r'\s*', lambda w: w, group_maps={'w': float})

tot_elec_info_patt = ProcessedPattern(r'Electrons, ' + r'\s*'.join([
    r'Alpha=\s*%s' % (int_patt % r'a'),
    r'Beta=\s*%s' % (int_patt % r'b'),
]) + r'\s*', 'elec', default_group_map=int)

act_elec_info_patt = ProcessedPattern(r'Active Electrons, ' + r'\s*'.join([
    r'Alpha=\s*%s' % (int_patt % r'a'),
    r'Beta=\s*%s' % (int_patt % r'b'),
]) + r'\s*', 'elec_act', default_group_map=int)

n_configs_patt = ProcessedPattern(r'\s*'.join([
    r'Alpha Strings=\s*%s' % (int_patt % r'a'),
    r'Beta Strings=\s*%s' % (int_patt % r'b'),
]) + r'\s*', 'config', default_group_map=int)


def read_mc_spec(file: TextIO, /, *, first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, mc_spec_start_patt, first_line=first_line,
                             err_msg='No MCSCF Specification is found')
    mc_spec = {}

    ref_type_info, line = search_in_file(file, ref_type_patt, first_line=line,
                                         err_msg='Could not find WF reference type')
    ci_type_info, line = search_in_file(file, ci_type_patt, first_line=line,
                                        err_msg='Could not find CI type')

    mc_patt = cas_patt if ci_type_info.is_cas else ras_patt
    active_spaces, line = search_in_file(file, mc_patt, first_line=line)
    mc_spec.update({'active_spaces': active_spaces, 'n_mo_act': sum(active_spaces)})

    mc_spec['mo_spaces'], line = search_in_file(file, ci_spaces_patt, first_line=line)

    match, line = search_in_file(file, sa_start_patt, first_line=line, n_skips=1)
    if match is not None:
        weights, line = findall_in_file(file, sa_weight_patt, max_matches=math.inf)
        mc_spec['sa_weights'] = np.asarray(weights)

    tot_elec_info, line = search_in_file(file, tot_elec_info_patt, first_line=line)
    act_elec_info, line = search_in_file(file, act_elec_info_patt, first_line=line)
    n_configs_info, line = search_in_file(file, n_configs_patt, first_line=line)

    mc_spec.update({
        'elec_act': act_elec_info, 'elec': tot_elec_info,
        'config': n_configs_info,
        'n_configs': reduce(lambda x, y: x * y, n_configs_info)
    })
    return mc_spec, line


mc_done_patt = re.compile(r'[CR]AS(SCF|CI) Done:')
state_patt = ProcessedPattern(r'\s*'.join([
    r'State:\s*%s' % (int_patt % r'state'),
    r'Energy \(Hartree\):\s*%s' % (float_patt % r'E'),
]), 'state_energy', group_maps={'state': int, 'E': float})

ci_complex_patt = r'\s*'.join((float_patt % 'real', float_patt % 'imag'))
ci_vec_patt = ProcessedPattern(
    r'\(\s*(?P<addr>\d*)\)\s*%s' % ci_complex_patt,
    'ci_coeff', match_funcs={'C': process_complex_match}, group_maps={'addr': int}
)


def read_ci_vecs(file: TextIO, n_configs: int, /, max_det: int = 50, *, first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, mc_done_patt, first_line=first_line, err_msg='No CI State information is found')

    # State energy info
    energy: list[float] = []
    state_idx: list[int] = []

    # CI Vectors data in COO format
    ci_data: list[complex | float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []

    while True:
        state_energy, line = search_in_file(file, state_patt,
                                            n_skips=1 if state_idx else 2, first_line=line)
        if not state_energy:
            break

        state_idx.append(state_energy.state)
        energy.append(state_energy.E)

        ci_vec_info, line = findall_in_file(file, ci_vec_patt, max_matches=max_det, max_skips=0)
        for state_info in ci_vec_info:
            ci_data.append(state_info.C)
            col_idx.append(state_info.addr - 1)
            row_idx.append(state_energy.state - 1)

    n_states = max(state_idx)
    ci_vecs = sparse.coo_array((ci_data, (row_idx, col_idx)), shape=(n_states, n_configs))

    df = pd.DataFrame({MCStates.ENERGY_COL: energy, MCStates.STATE_COL: state_idx})
    return dict(df_states=df, ci_vecs=ci_vecs, n_states=n_states), line


rdm_start_patt = re.compile(r'\*\* Printing Density Matrices for all States \*\*')
rdm_diag_patt = re.compile(r'For Simplicity: The diagonals of 1PDM for State:\s*(?P<idx>\d*)')


def read_rdm_diags(file: TextIO, n_states: int, n_mo_act: int, /, *, first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, rdm_start_patt, first_line=first_line, err_msg='No RDM information is found')

    rdm_diags: np.ndarray = np.empty((n_states, n_mo_act), dtype=np.float_)
    for i in range(n_states):
        rdm_diag, line = read_matrix_in_file(file, rdm_diag_patt, shape=(n_mo_act,))
        rdm_diags[i] = rdm_diag

    return dict(rdm_diags=rdm_diags), line


rdm_real_patt = re.compile('1PDM Matrix \(real\) :')
rdm_imag_patt = re.compile('1PDM Matrix \(imag\) :')


def read_rdms(file: TextIO, n_states: int, n_mo_act: int, /, first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, rdm_start_patt, first_line=first_line, err_msg='No RDM information is found')

    rdms: np.ndarray = np.zeros((n_states, n_mo_act, n_mo_act), dtype=np.complex128)
    for i in range(n_states):
        rdm_real, line = read_matrix_in_file(file, rdm_real_patt, shape=(n_mo_act, n_mo_act), first_line=line)
        rdm_imag, line = read_matrix_in_file(file, rdm_imag_patt, shape=(n_mo_act, n_mo_act), first_line=line)
        rdms[i] = rdm_real + rdm_imag * 1.j

    return {
        'rdms': rdms,
        'rdm_diags': np.diagonal(rdms, axis1=1, axis2=2)
    }, line


osc_patt = ProcessedPattern(
    r'Oscillator Strength For States\s*%s\s*:\s*%s\s*f=\s*%s' % (
        int_patt % r'initial',
        int_patt % r'final',
        float_patt % 'osc'
    ), 'osc_str', group_maps={'osc': float}, default_group_map=int
)


tdm_start_patt = ProcessedPattern(
    r'CI Transition Density Matrix\s*%s\s*:\s*%s\s*' % (
        int_patt % r'i',
        int_patt % r'f',
    ), 'tdm_info', default_group_map=int
)


def read_oscillator_strength(file: TextIO, n_states: int, n_ground: int, /, *,
                             first_line: str = '') -> tuple[ParsingResult, str]:
    _, line = search_in_file(file, tdm_start_patt.pattern, first_line=first_line, err_msg='No Oscillator information is found')

    n_pairs = n_ground * (2 * n_states - n_ground - 1) // 2
    osc_info, line = findall_in_file(file, osc_patt, max_matches=n_pairs, max_skips=3, first_line=line)

    initial_state: list[int] = []
    final_state: list[int] = []
    osc_strength: list[float] = []

    for d in osc_info:
        initial_state.append(d.initial)
        final_state.append(d.final)
        osc_strength.append(d.osc)

    df = pd.DataFrame({
        MCPeaks.INITIAL_STATE_COL: initial_state,
        MCPeaks.FINAL_STATE_COL: final_state,
        MCPeaks.OSC_COL: osc_strength
    })
    return dict(df_peaks=df), line


tdm_real_patt = re.compile('1TDM Matrix:\s*1')
tdm_imag_patt = re.compile('1TDM Matrix:\s*2')


def read_tdms(file: TextIO, /, *, first_line: str = '') -> tuple[ParsingResult, str]:
    tdms = {}
    initial_state: list[int] = []
    final_state: list[int] = []
    osc_strength: list[float] = []

    tdm_info, line = search_in_file(file, tdm_start_patt, first_line=first_line, err_msg='No TDM information is found')
    while tdm_info:
        tdm_real, line = read_matrix_in_file(file, tdm_real_patt, first_line=line)
        tdm_imag, line = read_matrix_in_file(file, tdm_imag_patt, first_line=line)
        tdms[(tdm_info.i, tdm_info.f)] = tdm_real + tdm_imag * 1.j

        osc_info, line = search_in_file(file, osc_patt, n_skips=0, first_line=line)
        initial_state.append(osc_info.initial)
        final_state.append(osc_info.final)
        osc_strength.append(osc_info.osc)

        tdm_info, line = search_in_file(file, tdm_start_patt, first_line=line)

    df = pd.DataFrame({
        MCPeaks.INITIAL_STATE_COL: initial_state,
        MCPeaks.FINAL_STATE_COL: final_state,
        MCPeaks.OSC_COL: osc_strength,
        'resource_idx': np.arange(len(tdms))
    })
    return dict(df_peaks=df, tdms=np.stack(list(tdms.values()))), line


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
        read_ci_vecs,
        read_rdm_diags,
        read_oscillator_strength
    ],
}

l910_parser_funcs_rdms: dict[str, list[Callable]] = {
    'l910': [
        read_mc_spec,
        read_ci_vecs,
        read_rdms,
        read_oscillator_strength,
    ],
}
