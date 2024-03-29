import re
from typing import AnyStr, TypeVar, ClassVar, TypeAlias

import attrs
import numpy as np
import scipy

from rich.progress import track

from mctools.cli.console import console
from mctools.core.cistring import DASGraph
from mctools.newcore.resource import Resource
from .base import MatrixParser, NewLinkParser
from ....core.filehandler import FileWithPosition
from ....core.parser.base import FWP
from ....core.pattern import ProcessedPattern
from ....core.stepper import LineStepper

__all__ = [
    'L910Parser',
]

R = TypeVar('R')
F: TypeAlias = FileWithPosition[AnyStr]


@attrs.define(eq=True, repr=True)
class L910Parser(NewLinkParser):
    START_ANCHOR: ClassVar[str] = 'l910.exe'

    STATE_ANCHOR: ClassVar[str] = 'Energy ('
    MAX_N_CONFIGS: ClassVar[int] = 50
    VEC_PATT: ClassVar[re.Pattern] = re.compile(
        r"""
        \(\s*(?P<addr>\d+)\s*\)\s*
        (?P<real>[+-]?\d+\.\d+)\s*
        (?P<imag>[+-]?\d+\.\d+)\s*
        """,
        flags=re.VERBOSE
    )

    SPIN_START_ANCHOR: ClassVar[str] = 'Computing Spin expectation values.'
    SPIN_ANCHOR: ClassVar[str] = '<Sx>='

    RDM_ANCHOR: ClassVar[str] = 'Final 1PDM for State:'
    RDM_REAL_ANCHOR: ClassVar[str] = '1PDM Matrix (real)'
    RDM_IMAG_ANCHOR: ClassVar[str] = '1PDM Matrix (imag)'

    TRANSITION_ANCHOR: ClassVar[str] = 'CI Transition Density Matrix'
    TDM_MATRIX_ANCHOR: ClassVar[str] = '1TDM Matrix:'
    OSC_ANCHOR: ClassVar[str] = 'Oscillator Strength'

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    ci_graph: DASGraph = attrs.field(init=False)
    n_states: int = attrs.field(init=False)
    n_initial: int = attrs.field(init=False)

    @ci_graph.default
    def _get_ci_space_default(self) -> DASGraph:
        iops = self.iops

        n_mos = iops[7]
        ras1_mo = iops[112]
        ras3_mo = iops[114]
        ras2_mo = n_mos - ras1_mo - ras3_mo
        # mcspace = MCSpace.from_ras_spec(
        #     (ras1_mo, ras2_mo, ras3_mo), iops[6],
        #     max_hole=iops[111], max_elec=iops[113],
        # )
        graph = DASGraph.from_ras_spec(
            (ras1_mo, ras2_mo, ras3_mo), iops[6],
            max_hole=iops[111], max_elec=iops[113]
        )

        return graph

    @n_states.default
    def _get_n_states_default(self) -> int:
        n_states = self.iops[17]
        n_states = n_states if n_states else 1
        if self.iops[13] in [0, 1]:
            n_states = sum(divmod(n_states, 10000))
        return n_states

    @n_initial.default
    def _get_n_initial_default(self) -> int:
        return self.iops[19]

    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[dict[Resource, np.ndarray], FWP[AnyStr]]:
        self.stepper.take(fwp)

        # Step to the beginning of the link
        start_in = self.stepper.get_anchor_predicate(self.START_ANCHOR)

        with console.status("Looking for Link 910..."):
            self.stepper.step_to(start_in, on_eof='raise')

        result: dict[Resource, np.ndarray] = {Resource.ci_graph: self.ci_graph}

        if (Resource.ci_energies | Resource.ci_vecs) & self.resources:
            result.update(self.read_energy_and_vector())

        if Resource.ci_spin & self.resources:
            result.update(self.read_spin_ev())

        if Resource.ci_int1e_rdms & self.resources:
            result.update(self.read_rdms())

        if Resource.ci_int1e_tdms & self.resources:
            result.update(self.read_osc_and_tdms())
        elif Resource.ci_osc in self.resources:
            result.update(self.read_osc())

        console.print('Finished parsing Link 910')
        return result, self.stepper.return_file()

    def read_energy_and_vector(self, /) -> dict[Resource, np.ndarray]:
        state_in = self.stepper.get_anchor_predicate(self.STATE_ANCHOR)

        state_idx = np.zeros(self.n_states, dtype=np.min_scalar_type(self.n_states))
        energies = np.zeros(self.n_states, dtype=np.float64)

        n_configs = min(self.ci_graph.n_configs, self.MAX_N_CONFIGS)
        vec_coef = np.zeros((self.n_states, n_configs), dtype=np.complex128)
        vec_addr = np.zeros((self.n_states, n_configs), dtype=np.min_scalar_type(self.ci_graph.n_configs))

        for idx in track(range(self.n_states), description="Reading CI Energies & Vectors..."):
            self.stepper.step_to(state_in, on_eof='raise')
            line = self.stepper.fwp.last_line.split()
            state_idx[idx] = line[1]
            energies[idx] = line[4]

            n_configs_read = 0
            while n_configs_read < n_configs:
                line = self.stepper.readline()
                for match in self.VEC_PATT.finditer(line):
                    vec_addr[n_configs_read] = int(match.group(1)) - 1
                    vec_coef[n_configs_read] = complex(
                        float(match.group(2)),
                        float(match.group(3))
                    )
                    n_configs_read += 1

        vectors = scipy.sparse.csr_matrix(
            (vec_coef.flat, (np.repeat(np.arange(self.n_states), n_configs), vec_addr.flat)),
            shape=(self.n_states, self.ci_graph.n_configs)
        )

        return {
            Resource.ci_state_idx: state_idx,
            Resource.ci_energies: energies,
            Resource.ci_vecs: vectors
        }

    def read_spin_ev(self) -> dict[Resource, np.ndarray]:
        spin_start_in = self.stepper.get_anchor_predicate(self.SPIN_START_ANCHOR)
        self.stepper.step_to(spin_start_in, on_eof='raise')

        spin = np.zeros(
            (self.n_states, 6),
            dtype='f4',
        )

        spin_in = self.stepper.get_anchor_predicate(self.SPIN_ANCHOR)
        for idx in track(range(self.n_states), description='Reading Spin expectation values...'):
            self.stepper.step_to(spin_in, on_eof='raise')
            line = self.stepper.fwp.last_line.split()
            spin[idx] = line[3:-3:2]

        return {Resource.ci_spin: spin}

    def read_rdms(self, /) -> dict[Resource, np.ndarray]:
        rdm_in = self.stepper.get_anchor_predicate(self.RDM_ANCHOR)
        rdm_parts_in = [
            self.stepper.get_anchor_predicate(anchor)
            for anchor in
            [self.RDM_REAL_ANCHOR, self.RDM_IMAG_ANCHOR]
        ]

        shape = (self.n_states, self.ci_graph.n_orb, self.ci_graph.n_orb)
        rdms = np.zeros(shape, dtype=[('R', 'f8'), ('I', 'f8')])

        matrix_parser = MatrixParser(stepper=self.stepper)
        for state_idx in track(range(self.n_states), description='Reading CI State RDMs...'):
            self.stepper.step_to(rdm_in)
            for component, rdm_part_in in zip(['R', 'I'], rdm_parts_in):
                self.stepper.step_to(rdm_part_in, on_eof='raise')
                matrix_parser.read_full_exact(rdms[state_idx][component])

        rdms = rdms.view('c16')
        return {Resource.ci_int1e_rdms: rdms}

    def read_osc(self, /) -> dict[Resource, np.ndarray]:
        osc_in = self.stepper.get_anchor_predicate(self.OSC_ANCHOR)
        osc = np.zeros(
            self.n_transitions,
            dtype=[('idx', 'u4'), ('fdx', 'u4'), ('osc', 'f8')]
        )
        for jdx in track(range(self.n_transitions), description='Reading CI Oscillator Strengths...'):
            self.stepper.step_to(osc_in, on_eof='raise')
            info = self.stepper.fwp.last_line.split()
            osc[jdx]['idx'] = info[4]
            osc[jdx]['fdx'] = info[6]
            osc[jdx]['osc'] = info[8]

        return {Resource.ci_initial_idx: osc['idx'],
                Resource.ci_final_idx: osc['fdx'],
                Resource.ci_osc: osc['osc']}

    def read_osc_and_tdms(self, /) -> dict[Resource, np.ndarray]:
        transition_in = self.stepper.get_anchor_predicate(self.TRANSITION_ANCHOR)
        osc_in = self.stepper.get_anchor_predicate(self.OSC_ANCHOR)
        tdm_in = self.stepper.get_anchor_predicate(self.TDM_MATRIX_ANCHOR)
        osc = np.zeros(
            self.n_transitions,
            dtype=[('idx', 'u4'), ('fdx', 'u4'), ('osc', 'f8')]
        )

        tdms_shape = (self.n_transitions, self.ci_graph.n_orb, self.ci_graph.n_orb)
        tdms = np.zeros(
            tdms_shape,
            dtype=[('R', 'f8'), ('I', 'f8')]
        )

        matrix_parser = MatrixParser(stepper=self.stepper)
        for jdx in track(range(self.n_transitions), description='Reading CI TDMs...'):
            self.stepper.step_to(transition_in)
            for component in ['R', 'I']:
                self.stepper.step_to(tdm_in)
                matrix_parser.read_full_exact(tdms[jdx][component])

            self.stepper.step_to(osc_in)
            info = self.stepper.fwp.last_line.split()
            osc[jdx]['idx'] = info[4]
            osc[jdx]['fdx'] = info[6]
            osc[jdx]['osc'] = info[8]

        tdms = tdms.view('c16')
        return {
            Resource.ci_initial_idx: osc['idx'],
            Resource.ci_final_idx: osc['fdx'],
            Resource.ci_osc: osc['osc'],
            Resource.ci_int1e_tdms: tdms
        }

    @property
    def n_transitions(self) -> int:
        return self.n_initial * (2 * self.n_states - self.n_initial - 1) // 2


def process_complex_match(
        match_dict: ProcessedPattern.MatchDict, /, *,
        real: str = 'real', imag: str = 'imag'
) -> complex:
    real = float(match_dict[real].replace('D', 'e'))
    imag = float(match_dict[imag].replace('D', 'e'))
    return real + imag * 1.j
