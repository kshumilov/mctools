from __future__ import annotations

from typing import ClassVar, AnyStr

import attrs
import numpy as np

from mctools.cli.console import console
from mctools.newcore.resource import Resource

from ....core import LineStepper
from ....core.parser.base import FWP

from .base import MatrixParser, NewLinkParser

__all__ = [
    'L302Parser',
]


@attrs.define(kw_only=True, eq=True, repr=True)
class L302Parser(NewLinkParser):
    START_ANCHOR: ClassVar[str] = 'l302.exe'
    NBASIS_ANCHOR: ClassVar[str] = 'NBasis'

    #     ANCHORS_X2C = (
    #         'Veff (p space)',
    #         'Trel (r space)',
    #         'Veff (r space)',
    #         'SO unc.',
    #         'DK / X2C integrals',
    #         'Orthogonalized basis functions'
    #     )

    RESOURCE_ANCHORS: ClassVar[Resource, str] = {
        Resource.ao_int1e_overlap: 'Overlap',
        Resource.ao_int1e_kinetic: 'Kinetic',
        Resource.ao_int1e_potential: 'Core Hamiltonian',
    }

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[dict[Resource, np.ndarray], FWP[AnyStr]]:
        self.stepper.take(fwp)

        # Step to the beginning of the link
        start_in = self.stepper.get_anchor_predicate(self.START_ANCHOR)

        with console.status('Looking for Link 302...'):
            self.stepper.step_to(start_in, on_eof='raise')

        n_aos = self.read_n_aos()
        result = self.read_stv_integrals(n_aos)

        console.print('Finished parsing Link 302')
        return result, self.stepper.return_file()

    def read_n_aos(self) -> int:
        nbasis_in = self.stepper.get_anchor_predicate(self.NBASIS_ANCHOR)
        self.stepper.step_to(nbasis_in, on_eof='raise')

        # '    NBasis = 575  MinDer = 0  MaxDer = 0\n'
        n_aos = int(self.stepper.fwp.last_line.strip().split()[2])
        return n_aos

    def read_stv_integrals(self, n_aos: int) -> dict[Resource, np.ndarray]:
        result: dict[Resource, np.ndarray] = {}
        matrix_parser = MatrixParser(stepper=self.stepper)

        console.print('Reading STV integrals...')
        for resource in self.resources:
            anchor_in = self.stepper.get_anchor_predicate(self.RESOURCE_ANCHORS[resource])

            with console.status(f'Reading {resource.name}...'):
                self.stepper.step_to(anchor_in, on_eof='raise')
                matrix = result.setdefault(resource, np.zeros((n_aos, n_aos), dtype=np.float32))
                matrix_parser.read_tril_exact(matrix)
                matrix += (matrix.T - np.diag(np.diag(matrix)))
        return result
