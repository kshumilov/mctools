from __future__ import annotations

import re

from enum import Flag, auto
from functools import reduce
from typing import TextIO, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd

from scipy import sparse

from parsing.gaussian.log.route.route import Link, RouteLine
from parsing.gaussian.log.utils import read_matrix_in_file
from parsing.core import (
    ProcessedPattern,

)
from parsing.core.pattern import simple_int_tmplt, int_patt, float_patt, bool_patt, ParsingResultType
from parsing.oldlib import findall_in_file
from parsing import search_in_file
from core.cistring import RASMOs, Electrons
from core.mcspace import MCSpace, MOSpacePartition
from core.mcstates import MCStates
from core.mctransitions import MCTransitions

if TYPE_CHECKING:
    from parsing.gaussian.log import Route

__all__ = [
    'L910CalcProperties',

    'l910_parser_funcs_fast',
    'l910_parser_funcs_all',

    'read_rdms',
    'read_mc_spec',
    'read_states',
    'read_rdm_diags',
    'read_oscillator_strength',

    'rdm_start_patt', 'rdm_real_patt', 'rdm_imag_patt',
    'tdm_real_patt', 'tdm_imag_patt',
]


mc_spec_start_patt = re.compile(r'Input Summary:')


def find_mc_spec(file: TextIO, /, *, first_line: str = ''):
    _, line = search_in_file(file, mc_spec_start_patt, first_line=first_line,
                             err_msg='No MCSCF Specification is found')
    return {}, line


def bool_map(x: str) -> bool:
    return x.capitalize() == 'T'


ref_type_patt = ProcessedPattern(
    '\s*,\s*'.join([
        r'RHF=\s*%s' % (bool_patt % 'is_rhf'),
        r'CRHF=\s*%s' % (bool_patt % 'is_crhf'),
        r'ROHF=\s*%s' % (bool_patt % 'is_rohf'),
        r'GHF=\s*%s' % (bool_patt % 'is_ghf')
    ]),
    constructor='ref_type',
    default_group_map=bool_map
)


def read_wf_ref_type(file: TextIO, /, *, first_line: str = '',) -> tuple[ParsingResultType, str]:
    ref_type_info, line = search_in_file(file, ref_type_patt, first_line=first_line,
                                         err_msg='Could not find WF reference type')
    return {'wf_ref_type': ref_type_info}, line


ci_type_patt = ProcessedPattern(
    r'\s*'.join([
        r'CAS=\s*%s\s*,' % (bool_patt % r'is_cas'),
        r'RAS=\s*%s\s*,' % (bool_patt % r'is_ras'),
        r'MRCISD=\s*%s\s*' % (bool_patt % r'is_mrcisd'),
    ]),
    constructor='ci_type',
    default_group_map=bool_map
)

ras_patt = ProcessedPattern(
    r'RAS\(\s*%s,\s*%s,\s*%s\)' % (
        (int_patt % r'r1'),
        (int_patt % r'r2'),
        (int_patt % r'r3'),
    ),
    constructor=RASMOs,
    default_group_map=int
)

cas_patt = ProcessedPattern(
    r'CAS\(\s*%se\s*,\s*%so\s*\)' % (
        (int_patt % r'n_e'),
        (int_patt % r'n_o'),
    ),
    constructor=lambda n_e, n_o: RASMOs(0, n_o, 0),
    default_group_map=int
)


def read_ci_type(file: TextIO, /, *, first_line: str = '',) -> tuple[ParsingResultType, str]:
    ci_type_info, line = search_in_file(file, ci_type_patt, first_line=first_line,
                                        err_msg='Could not find CI type')

    mc_patt = cas_patt if ci_type_info.is_cas else ras_patt
    active_spaces, line = search_in_file(file, mc_patt, first_line=line,
                                         err_msg='Could not find Active Spaces definition')
    result = {'active_spaces': active_spaces, 'n_mo_act': lambda _: sum(active_spaces)}
    return result, line


mo_space_partition_patt = ProcessedPattern(
    r'\s*'.join([
        r'NTOrb=\s*%s' % simple_int_tmplt,
        r'NIOrb=\s*%s' % (int_patt % r'i'),
        r'NAOrb=\s*%s' % (int_patt % r'a'),
        r'NVOrb=\s*%s' % (int_patt % r'v'),
    ]) + r'\s*',
    constructor=lambda i, a, v: MOSpacePartition.from_spaces(a, i, v),
    default_group_map=int
)


def read_mo_partition(file: TextIO, /, *, first_line: str = '',) -> tuple[ParsingResultType, str]:
    mo_space_partition, line = search_in_file(file, mo_space_partition_patt, first_line=first_line,
                                              err_msg='Could not find MO Partition information')
    return {'mo_space_partition': mo_space_partition}, line


sa_start_patt = re.compile(r'SA Weights Read:')
sa_weight_patt = ProcessedPattern(
    r'\s*'.join([
        r'State:\s*%s' % simple_int_tmplt,
        r'Weight:\s*%s' % (float_patt % r'w'),
    ]) + r'\s*',
    constructor=lambda w: w,
    group_maps={'w': float}
)


def read_sa_weights(file: TextIO, n_states: int, /, *, first_line: str = '',) -> tuple[ParsingResultType, str]:
    search_in_file(file, sa_start_patt, first_line=first_line, n_skips=1, err_msg='Could not find SA-weights')
    weights, line = findall_in_file(file, sa_weight_patt, max_matches=n_states)
    return {'sa_weights': np.asarray(weights)}, line


tot_elec_info_patt = ProcessedPattern(
    r'Electrons, ' + r'\s*'.join([
        r'Alpha=\s*%s' % (int_patt % r'alpha'),
        r'Beta=\s*%s' % (int_patt % r'beta'),
    ]) + r'\s*',
    constructor=Electrons,
    default_group_map=int
)

act_elec_info_patt = ProcessedPattern(
    r'Active Electrons, ' + r'\s*'.join([
        r'Alpha=\s*%s' % (int_patt % r'alpha'),
        r'Beta=\s*%s' % (int_patt % r'beta'),
    ]) + r'\s*',
    constructor=Electrons,
    default_group_map=int
)


def read_elec_info(file: TextIO, /, *, first_line: str = '') -> tuple[ParsingResultType, str]:
    tot_elec_info, line = search_in_file(file, tot_elec_info_patt, first_line=first_line)
    act_elec_info, line = search_in_file(file, act_elec_info_patt, first_line=line)
    return {
        'elec_act': act_elec_info, 'elec': tot_elec_info,
    }, line


n_configs_patt = ProcessedPattern(
    r'\s*'.join([
        r'Alpha Strings=\s*%s' % (int_patt % r'alpha'),
        r'Beta Strings=\s*%s' % (int_patt % r'beta'),
    ]) + r'\s*',
    constructor='config',
    default_group_map=int
)


def read_config_info(file: TextIO, /, *, first_line: str = '') -> tuple[ParsingResultType, str]:
    n_configs_info, line = search_in_file(file, n_configs_patt, first_line=first_line)
    return {
        'config': n_configs_info,
        'n_configs': reduce(lambda x, y: x * y, n_configs_info)
    }, line


def read_mc_spec(file: TextIO, /, *, first_line: str = '',) -> tuple[ParsingResultType, str]:
    *_, line = find_mc_spec(file, first_line=first_line)
    ref_type_info, line = read_wf_ref_type(file, first_line=line)
    ci_type_info, line = read_ci_type(file, first_line=line)
    mo_space_partition_info, line = read_mo_partition(file, first_line=line)

    match, line = search_in_file(file, sa_start_patt, first_line=line, n_skips=1)
    if match is not None:
        weights, line = findall_in_file(file, sa_weight_patt, max_matches=float('+inf'))
        sa_info = {'sa_weights': np.asarray(weights)}
    else:
        sa_info = {}

    elec_info, line = read_elec_info(file, first_line=line)
    config_info, line = read_config_info(file, first_line=line)
    return {
        **ref_type_info,
        **ci_type_info,
        **mo_space_partition_info,
        **sa_info, **elec_info,
        **config_info
    }, line


mc_done_patt = re.compile(r'[CR]AS(SCF|CI) Done:')
state_patt = ProcessedPattern(
    r'\s*'.join([
        r'State:\s*%s' % (int_patt % r'state'),
        r'Energy \(Hartree\):\s*%s' % (float_patt % r'E'),
    ]),
    'state_energy',
    group_maps={'state': int, 'E': float}
)


def process_complex_match(match_dict: ProcessedPattern.MatchDict, *, real: str = 'real', imag: str = 'imag') -> complex:
    real = float(match_dict[real].replace('D', 'e'))
    imag = float(match_dict[imag].replace('D', 'e'))
    return real + imag * 1.j


ci_vec_patt = ProcessedPattern(
    r'\(\s*(?P<addr>\d*)\)\s*%s' % r'\s*'.join((float_patt % 'real', float_patt % 'imag')),
    'ci_coeff', match_funcs={'C': process_complex_match}, group_maps={'addr': int}
)


def read_states(file: TextIO, n_configs: int, /, max_det: int = 50, *, first_line: str = '') -> tuple[ParsingResultType, str]:
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
            ci_data.append(state_info.CalcT)
            col_idx.append(state_info.addr - 1)
            row_idx.append(state_energy.state - 1)

    n_states = max(state_idx)
    ci_vecs = sparse.coo_array((ci_data, (row_idx, col_idx)), shape=(n_states, n_configs))

    df = pd.DataFrame({MCStates.ENERGY_COL: energy, MCStates.STATE_COL: state_idx})
    return dict(df_states=df, ci_vecs=ci_vecs, n_states=n_states), line


rdm_start_patt = re.compile(r'\*\* Printing Density Matrices for all States \*\*')
rdm_diag_patt = re.compile(r'For Simplicity: The diagonals of 1PDM for State:\s*(?P<idx>\d*)')


def read_rdm_diags(file: TextIO, n_states: int, n_mo_act: int, /, *, first_line: str = '') -> tuple[ParsingResultType, str]:
    _, line = search_in_file(file, rdm_start_patt, first_line=first_line, err_msg='No RDM information is found')

    rdm_diags: np.ndarray = np.empty((n_states, n_mo_act), dtype=np.float_)
    for i in range(n_states):
        rdm_diag, line = read_matrix_in_file(file, rdm_diag_patt, shape=(n_mo_act,))
        rdm_diags[i] = rdm_diag

    return dict(rdm_diags=rdm_diags), line


rdm_real_patt = re.compile('1PDM Matrix \(real\) :')
rdm_imag_patt = re.compile('1PDM Matrix \(imag\) :')


def read_rdms(file: TextIO, n_states: int, n_mo_act: int, /, first_line: str = '') -> tuple[ParsingResultType, str]:
    _, line = search_in_file(file, rdm_start_patt, first_line=first_line, err_msg='No RDM information is found')

    rdms: np.ndarray = np.zeros((n_states, n_mo_act, n_mo_act), dtype=np.complex128)
    for i in range(n_states):
        rdm_real, line = read_matrix_in_file(file, rdm_real_patt, shape=(n_mo_act, n_mo_act), first_line=line)
        rdm_imag, line = read_matrix_in_file(file, rdm_imag_patt, shape=(n_mo_act, n_mo_act), first_line=line)
        rdms[i] = rdm_real + rdm_imag * 1.j

    return {
        'rdms': rdms,
        'rdm_diags': np.abs(np.diagonal(rdms, axis1=1, axis2=2))
    }, line


osc_patt = ProcessedPattern(
    r'Oscillator Strength For States\s*%s\s*:\s*%s\s*f=\s*%s' % (
        int_patt % r'initial',
        int_patt % r'final',
        float_patt % 'osc'
    ),
    constructor='osc_str',
    group_maps={'osc': float},
    default_group_map=int
)


tdm_start_patt = ProcessedPattern(
    r'CI Transition Density Matrix\s*%s\s*:\s*%s\s*' % (
        int_patt % r'i',
        int_patt % r'f',
    ),
    constructor='tdm_info',
    default_group_map=int
)


def read_oscillator_strength(file: TextIO, n_states: int, n_ground: int, /, *,
                             first_line: str = '') -> tuple[ParsingResultType, str]:
    _, line = search_in_file(file, tdm_start_patt.pattern, first_line=first_line,
                             err_msg='No Oscillator information is found')

    n_pairs = n_ground * (2 * n_states - n_ground - 1) // 2
    osc_info, line = findall_in_file(file, osc_patt, max_matches=n_pairs, max_skips=-1, first_line=line)

    initial_state: list[int] = []
    final_state: list[int] = []
    osc_strength: list[float] = []

    for d in osc_info:
        initial_state.append(d.initial)
        final_state.append(d.final)
        osc_strength.append(d.osc)

    df = pd.DataFrame({
        MCTransitions.INITIAL_STATE_COL: initial_state,
        MCTransitions.FINAL_STATE_COL: final_state,
        MCTransitions.OSC_COL: osc_strength
    })
    return dict(df_peaks=df), line


tdm_real_patt = re.compile('1TDM Matrix:\s*1')
tdm_imag_patt = re.compile('1TDM Matrix:\s*2')


def read_tdms(file: TextIO, /, *, first_line: str = '') -> tuple[ParsingResultType, str]:
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
        MCTransitions.INITIAL_STATE_COL: initial_state,
        MCTransitions.FINAL_STATE_COL: final_state,
        MCTransitions.OSC_COL: osc_strength,
        MCTransitions.RESOURCE_COL: np.arange(len(tdms))
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


def parse_l910_route(route_line: RouteLine) -> tuple[ParsingResultType, Flag]:
    if Link.L910.value not in route_line.links:
        raise ValueError(f'{route_line!r} does not have link {Link.L910}')

    calc_properties = L910CalcProperties(0)

    # TODO Return dict of parameters to create MCSpace, don't create it here
    n_mos = route_line.get(7)
    ras1_mo = route_line.get(112)
    ras3_mo = route_line.get(114)
    ras2_mo = n_mos - ras1_mo - ras3_mo
    mcspace = MCSpace.from_space_spec(
        (ras1_mo, ras2_mo, ras3_mo), route_line.get(6),
        max_hole=route_line.get(111), max_elec=route_line.get(113)
    )
    del n_mos, ras1_mo, ras2_mo, ras3_mo

    match route_line.get(13):
        case 0 | 1:
            calc_properties |= L910CalcProperties.performs_davidson

    # Account for M0000, where M is number higher energy states in Davidson
    n_states = route_line.get(17)
    n_states = n_states if n_states else 1
    if L910CalcProperties.performs_davidson in calc_properties:
        n_states = n_states // 10000 + n_states % 10000

    if route_line.get(18) > 0:
        calc_properties |= L910CalcProperties.performs_state_averaging

    n_ground = route_line.get(19)
    n_ground = n_states if n_ground < 0 else n_ground
    if n_ground:
        calc_properties |= L910CalcProperties.calculates_osc_strength

    match route_line.get(32, 1):
        case 0 | 1:
            calc_properties |= L910CalcProperties.calculates_rdms

    if route_line.get(54) > 0:
        calc_properties |= L910CalcProperties.calculates_spin_ev

    if bool(route_line.get(131)):
        calc_properties |= L910CalcProperties.calculates_tdms

    return {
        'mcspace': mcspace,
        'n_states': n_states,
        'n_ground': n_ground,

        'n_configs': lambda _: mcspace.n_configs,
        'n_mo_act': lambda _: mcspace.n_mo_act
    }, calc_properties


class L910ParserProperties(Flag):
    parse_mc_config = auto()
    parse_states = auto()
    parse_spin_ev = auto()
    parse_rdms = auto()
    parse_rdm_diags = auto()
    parse_osc_strength = auto()
    parse_tdms = auto()

    @classmethod
    def default(cls) -> 'L910ParserProperties':
        return cls.parse_states | cls.parse_states | cls.parse_rdm_diags


class L910CalcProperties(Flag):
    calculates_spin_ev = auto()  # Spin Expectation Value
    calculates_rdms = auto()
    calculates_osc_strength = auto()
    calculates_tdms = auto()

    performs_state_averaging = auto()
    performs_davidson = auto()


def build_l910_parser_funcs(route: Route,
                            properties: L910ParserProperties | None = None,
                            result: ParsingResultType | None = None) -> dict[str, list[Callable]]:
    if not Link.L910 in route:
        return {Link.L910.value: []}

    route_line = route.get_route_line(Link.L910)
    route_result, calc_properties = parse_l910_route(route_line)

    result = result if result is not None else {}
    result |= route_result

    parser_properties = properties if properties is not None else L910ParserProperties.default()

    parser_funcs: list[callable] = []
    if L910ParserProperties.parse_mc_config in parser_properties:
        parser_funcs.extend([find_mc_spec, read_wf_ref_type])
        if 'mcspace' not in result:
            parser_funcs.append(read_ci_type)

        if 'mo_space_partition' not in result:
            parser_funcs.append(read_mo_partition)

        if L910CalcProperties.performs_state_averaging in calc_properties:
            parser_funcs.append(read_sa_weights)

        if 'n_configs' not in result:
            parser_funcs.extend([read_elec_info, read_config_info])

        if 'n_states' in result:
            parser_funcs.append(read_states)

    if L910ParserProperties.parse_states in parser_properties:
        parser_funcs.append(read_states)

    if L910CalcProperties.calculates_rdms in calc_properties:
        if L910ParserProperties.parse_rdms in parser_properties:
            parser_funcs.append(read_rdms)
        elif L910ParserProperties.parse_rdm_diags in parser_properties:
            parser_funcs.append(read_rdm_diags)

    if L910CalcProperties.calculates_spin_ev:
        pass

    if L910CalcProperties.calculates_osc_strength:
        if L910CalcProperties.calculates_tdms and L910ParserProperties.parse_rdms in parser_properties:
            parser_funcs.append(read_tdms)
        else:
            parser_funcs.append(read_oscillator_strength)

    return {Link.L910.name: parser_funcs}


l910_parser_funcs_fast: dict[str, list[Callable]] = {
    'l910': [
        read_mc_spec,
        read_states,
        read_rdm_diags,
        read_oscillator_strength
    ],
}

l910_parser_funcs_all: dict[str, list[Callable]] = {
    'l910': [
        read_mc_spec,
        read_states,
        read_rdms,
        read_tdms,
    ],
}
