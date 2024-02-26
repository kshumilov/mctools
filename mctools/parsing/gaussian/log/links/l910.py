import re
from typing import AnyStr, TypeVar, ClassVar, TypeAlias

import attrs
import numpy as np
from tqdm import tqdm

from mctools.core.mcspace import MCSpace
from mctools.core.resource import Resource

from ....core.parser.base import FWP
from ....core.pattern import ProcessedPattern
from ....core.stepper import LineStepper
from ....core.filehandler import FileWithPosition

from .base import MatrixParser, NewLinkParser


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

    RDM_ANCHOR: ClassVar[str] = 'Final 1PDM for State:'
    RDM_REAL_ANCHOR: ClassVar[str] = '1PDM Matrix (real)'
    RDM_IMAG_ANCHOR: ClassVar[str] = '1PDM Matrix (imag)'

    OSC_ANCHOR: ClassVar[str] = 'Oscillator Strength'

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    ci_space: MCSpace = attrs.field(init=False)
    n_states: int = attrs.field(init=False)
    n_initial: int = attrs.field(init=False)

    @ci_space.default
    def _get_ci_space_default(self) -> MCSpace:
        iops = self.iops

        n_mos = iops[7]
        ras1_mo = iops[112]
        ras3_mo = iops[114]
        ras2_mo = n_mos - ras1_mo - ras3_mo
        mcspace = MCSpace.from_space_spec(
            (ras1_mo, ras2_mo, ras3_mo), iops[6],
            max_hole=iops[111], max_elec=iops[113],
        )

        return mcspace

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
        self.stepper.step_to(start_in, on_eof='raise')

        result: dict[Resource, np.ndarray] = {Resource.ci_space: self.ci_space}

        if (Resource.ci_energy | Resource.ci_vecs) & self.resources:
            result.update(self.read_states())

        if Resource.ci_int1e_rdms & self.resources:
            result.update(self.read_rdms())

        if Resource.ci_osc & self.resources:
            result.update(self.read_osc())

        return result, self.stepper.return_file()

    def read_states(self, /) -> dict[Resource, np.ndarray]:
        state_in = self.stepper.get_anchor_predicate(self.STATE_ANCHOR)

        states = np.zeros(self.n_states, dtype=[('idx', 'u4'), ('energy', 'f4')])

        config_dtype = np.dtype([('addr', 'u4'), ('C', 'c8')])

        n_configs = min(self.ci_space.n_configs, self.MAX_N_CONFIGS)
        vectors = np.zeros((self.n_states, n_configs), dtype=config_dtype)

        for idx in tqdm(range(self.n_states), unit='State'):
            self.stepper.step_to(state_in)
            line = self.stepper.fwp.last_line.split()
            states[idx]['idx'] = line[1]
            states[idx]['energy'] = line[4]

            n_configs_read = 0
            while n_configs_read < n_configs:
                vector = vectors[idx]
                line = self.stepper.readline()
                for match in self.VEC_PATT.finditer(line):
                    vector[n_configs_read]['addr'] = int(match.group(1)) - 1
                    vector[n_configs_read]['C'] = complex(
                        float(match.group(2)),
                        float(match.group(3))
                    )
                    n_configs_read += 1

        return {Resource.ci_energy: states, Resource.ci_vecs: vectors}

    def read_rdms(self, /) -> dict[Resource, np.ndarray]:
        rdm_in = self.stepper.get_anchor_predicate(self.RDM_ANCHOR)
        rdm_parts_in = [
            self.stepper.get_anchor_predicate(anchor)
            for anchor in
            [self.RDM_REAL_ANCHOR, self.RDM_IMAG_ANCHOR]
        ]

        shape = (self.n_states, self.ci_space.n_mo_act, self.ci_space.n_mo_act)
        rdms = np.zeros(shape, dtype=[('R', 'f4'), ('I', 'f4')])

        matrix_parser = MatrixParser(stepper=self.stepper)
        for state_idx in tqdm(range(self.n_states), unit='RDM'):
            self.stepper.step_to(rdm_in)
            for component, rdm_part_in in zip(['R', 'I'], rdm_parts_in):
                self.stepper.step_to(rdm_part_in)
                matrix_parser.read_full_exact(rdms[state_idx][component])

        rdms = rdms.view('c8')
        return {Resource.ci_int1e_rdms: rdms}

    def read_osc(self, /) -> dict[Resource, np.ndarray]:
        osc_in = self.stepper.get_anchor_predicate(self.OSC_ANCHOR)

        n_transitions = self.n_initial * (2 * self.n_states - self.n_initial - 1) // 2
        transitions = np.zeros(
            n_transitions,
            dtype=[('idx', 'u4'), ('fdx', 'u4'), ('osc', 'f4')]
        )

        for jdx in tqdm(range(n_transitions), unit='transition'):
            self.stepper.step_to(osc_in)
            info = self.stepper.fwp.last_line.split()
            transitions[jdx]['idx'] = info[4]
            transitions[jdx]['fdx'] = info[6]
            transitions[jdx]['osc'] = info[8]

        return {Resource.ci_osc: transitions}


def process_complex_match(
        match_dict: ProcessedPattern.MatchDict, /, *,
        real: str = 'real', imag: str = 'imag'
) -> complex:
    real = float(match_dict[real].replace('D', 'e'))
    imag = float(match_dict[imag].replace('D', 'e'))
    return real + imag * 1.j