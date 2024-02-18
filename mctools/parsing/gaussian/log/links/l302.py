from __future__ import annotations

from core import Resources
from parsing.core import Task, ListenerConfig
from .parser import LinkParser

__all__ = [
    'L302Parser',
]


class L302Parser(LinkParser):
    ANCHORS_STV = ('Overlap', 'Kinetic', 'Core Hamiltonian')
    ANCHORS_X2C = ('Veff (p space)', 'Trel (r space)', 'Veff (r space)',
                   'SO unc.', 'DK / X2C integrals',
                   'Orthogonalized basis functions')

    DEFAULT_RESOURCES = Resources.ao_int1e_stv
    DEFAULT_TASKS = {
        resource: Task(
            anchor=anchor,
            handle='read_square_matrix',
            settings=ListenerConfig(dispatch_file=False)
        )
        for resource, anchor in zip(Resources.ao_int1e_stv, ANCHORS_STV)
    }

    def parse_iops(self, requested_resources: Resources, /) -> Resources:
        if self.iops.get(33) in (1, 5):
            requested_resources &= Resources.ao_int1e_stv
        else:
            requested_resources ^= Resources.ao_int1e_stv
        return requested_resources
