import re
from typing import AnyStr, TypeVar, ClassVar, TypeAlias

import attrs
import numpy as np
from tqdm import tqdm

from core.mcspace import MCSpace
from core.resource import Resource

from parsing.core.parser.base import Parser
from parsing.core.parser.dict import Listener
from parsing.core.pattern import ProcessedPattern
from parsing.core.stepper import LineStepper
from parsing.core.filehandler import FileWithPosition

from parsing.gaussian.log.links.base import LinkParser, RealMatrixParser, MatrixSymmetry


def process_complex_match(
        match_dict: ProcessedPattern.MatchDict, /, *,
        real: str = 'real', imag: str = 'imag'
) -> complex:
    real = float(match_dict[real].replace('D', 'e'))
    imag = float(match_dict[imag].replace('D', 'e'))
    return real + imag * 1.j


R = TypeVar('R')
F: TypeAlias = FileWithPosition[AnyStr]


@attrs.define(repr=True)
class CIStateParser(Parser):
    STATE_ANCHOR: ClassVar[str] = 'Energy ('
    MAX_N_CONFIGS: ClassVar[int] = 50

    ci_vec_patt: ClassVar[re.Pattern] = re.compile(
        r"""
        \(\s*(?P<addr>\d+)\s*\)\s*
        (?P<real>[+-]?\d+\.\d+)\s*
        (?P<imag>[+-]?\d+\.\d+)\s*
        """,
        flags=re.VERBOSE
    )

    n_states: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
    )
    n_configs: int = attrs.field(
        default=0,
        converter=lambda x: min(x, CIStateParser.MAX_N_CONFIGS),
        validator=attrs.validators.ge(0),
    )

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    def parse_file(self, fwp: FileWithPosition, /) -> tuple[R, FileWithPosition]:
        self.stepper.take(fwp)
        state_in = self.stepper.get_anchor_predicate(self.STATE_ANCHOR)

        # coeff_dtype = np.dtype([('real', 'f4'), ('imag', 'f4')])
        config_dtype = np.dtype([('addr', 'u4'), ('C', 'c8')])
        state_dtype = np.dtype([
            ('idx', 'u4'),
            ('energy', 'f4'),
            ('vec', config_dtype, self.n_configs)
        ])
        states = np.zeros(self.n_states, dtype=state_dtype)

        for idx in tqdm(range(self.n_states), unit='State'):
            state = states[idx]

            self.stepper.step_to(state_in)
            line = self.stepper.fwp.last_line.split()
            state['idx'] = line[1]
            state['energy'] = line[4]

            n_configs_read = 0
            while n_configs_read < self.n_configs:
                line = self.stepper.readline()
                for match in self.ci_vec_patt.finditer(line):
                    state['vec'][n_configs_read]['addr'] = int(match.group(1)) - 1
                    state['vec'][n_configs_read]['C'] = complex(
                        float(match.group(2)), float(match.group(3)))
                    n_configs_read += 1

        return states, self.stepper.return_file()


@attrs.define(repr=True)
class CIRDMParser(RealMatrixParser):
    RDM_ANCHOR: ClassVar[str] = 'Final 1PDM for State:'
    RDM_REAL_ANCHOR: ClassVar[str] = '1PDM Matrix (real)'
    RDM_IMAG_ANCHOR: ClassVar[str] = '1PDM Matrix (imag)'

    n_states: int = attrs.field(
        default=0,
        validator=attrs.validators.instance_of(int),
    )

    def parse_file(self, fwp: F, /) -> tuple[R, F]:
        self.stepper.take(fwp)

        rdm_in = self.stepper.get_anchor_predicate(self.RDM_ANCHOR)
        rdm_parts_in = [
            self.stepper.get_anchor_predicate(anchor)
            for anchor in
            [self.RDM_REAL_ANCHOR, self.RDM_IMAG_ANCHOR]
        ]

        shape = (self.n_states, self.n_rows, self.n_cols)
        rdms = np.zeros(shape, dtype=[('R', 'f4'), ('I', 'f4')])

        for state_idx in tqdm(range(self.n_states), unit='RDM'):
            self.stepper.step_to(rdm_in)
            for component, rdm_part_in in zip(['R', 'I'], rdm_parts_in):
                self.stepper.step_to(rdm_part_in)
                self.read_matrix(rdms[state_idx][component])

        rdms = rdms.view('c8')
        return rdms, self.stepper.return_file()


@attrs.define(repr=True)
class CIOSCParser(Parser):
    OSC_ANCHOR: ClassVar[str] = 'Oscillator Strength'

    n_states: int = attrs.field(
        default=0,
        validator=attrs.validators.instance_of(int),
    )

    n_initial: int = attrs.field(
        default=0,
        validator=attrs.validators.instance_of(int),
    )

    stepper: LineStepper = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    def parse_file(self, fwp: F, /) -> tuple[R, F]:
        self.stepper.take(fwp)
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

        return transitions, self.stepper.return_file()


@attrs.define(eq=True, repr=True)
class L910Parser(LinkParser):
    PARSABLE_RESOURCES: ClassVar[Resource] = (
            Resource.ci_energy |
            Resource.ci_int1e_rdms |
            Resource.ci_osc |
            Resource.ci_space
    )

    DEFAULT_LISTENERS = {
        Resource.ci_energy: Listener(
            parser=CIStateParser(),
            label=Resource.ci_energy,
            anchor='Done',
            max_runs=1,
        ),
        Resource.ci_int1e_rdms: Listener(
            parser=CIRDMParser(),
            label=Resource.ci_int1e_rdms,
            anchor='** Printing Density Matrices for all States **',
            max_runs=1,
        ),
        Resource.ci_osc: Listener(
            parser=CIOSCParser(),
            label=Resource.ci_osc,
            anchor='Using Dipole Ints in file',
            max_runs=1,
        )
    }

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

    def postprocess(self, raw_data: dict[Resource, list[np.ndarray]]) -> tuple[dict[Resource, np.ndarray | MCSpace], F]:
        raw_data = super().postprocess(raw_data)

        result: dict[Resource, np.ndarray | MCSpace] = {}
        for resource, resource_result in raw_data.items():
            result[resource] = resource_result.pop()

        result[Resource.ci_space] = self.ci_space
        return result

    def update_listener(self, listener: Listener) -> None:
        match listener.parser:
            case CIStateParser():
                listener.parser = attrs.evolve(
                    listener.parser,
                    n_states=self.n_states,
                    n_configs=self.ci_space.n_configs,
                )
            case CIRDMParser():
                listener.parser = attrs.evolve(
                    listener.parser,
                    n_states=self.n_states,
                    n_rows=self.ci_space.n_mo_act,
                    n_cols=self.ci_space.n_mo_act,
                    symmetry=MatrixSymmetry.FULL,
                )
            case CIOSCParser():
                listener.parser = attrs.evolve(
                    listener.parser,
                    n_states=self.n_states,
                    n_initial=self.n_initial,
                )
