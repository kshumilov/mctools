from __future__ import annotations


import abc
from graphlib import TopologicalSorter

from typing import TypeVar, Generic, Type, Protocol, Any, Sequence, Self

import attrs

T = TypeVar('T')
U = TypeVar('U')


@attrs.frozen(repr=True, eq=True, hash=True, frozen=True)
class Dependency(Generic[T], metaclass=abc.ABCMeta):
    obj_type: Type[T]
    predecessors: Sequence[Dependency[T]] = attrs.field(factory=tuple)
    is_optional: bool = False

    @property
    def can_be_resolved(self) -> bool:
        return all(d.can_be_resolved for d in self.predecessors)

    @abc.abstractmethod
    def resolve(self, *dependencies: Dependency[T]) -> T:
        raise NotImplementedError


class Storable(Protocol):
    def to_hdf5(self, filename: str, prefix: str) -> None:
        ...

    @classmethod
    def from_hdf5(cls, filename: str, prefix: str) -> Storable:
        ...


class Resource(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def build(cls, *arg: Any, **kwarg: Any) -> Resource:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_dependencies(cls) -> Sequence[Dependency[Resource]]:
        raise NotImplementedError


class Consolidator(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def analyze(self, *calculators: Sequence[Analyzer[Self]]) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def set(self, resource: Resource) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def get_resource_dependencies(cls) -> Sequence[Resource]:
        raise NotImplementedError


C = TypeVar('C', bound=Consolidator)
R = TypeVar('R')


class Analyzer(Protocol[C]):
    def __cal__(self, consolidator: C) -> Any:
        ...


build_dependencies = {
    'mcwfn@': {
        'states@': {
            'state_i': (
                {
                    'energy!': {},
                    'spin?': {
                        'Sx!': {},
                        'Sy!': {},
                        'Sz!': {},
                        'Sx^2!': {},
                        'Sy^2!': {},
                        'Sz^z!': {},
                        'S^2!': {'Sx^2', 'Sy^2', 'Sz^z'},
                    },
                    'rdm*?': {
                        'n_mo_act',
                    },
                    'vec*': {
                        'n_configs',
                    }
                },
                'n_states',
            )
        },
        'transitions@': {
            'transition_i': (
                {
                    'idx!': {},
                    'fdx!': {},
                    'osc!': {'tdm'},
                    'tdm*?': {
                        'n_mo_act',
                    },
                },
                'n_transitions'
            ),
        },
        'ci_space@': {
            'active_spaces': {},
            'occ_restrictions': {},
            'mo_space?': {},
        },
        'mo_basis?': {},
        'n_states': {'ci_space'},
    },
    'mo_basis': {
        ('mo', 'n_mos'): {
            'coeff': {
                'mo_ansatz': {},
                'ao_basis.n_aos': {},
            }
        },
        'ao_basis': {
            ('ao', 'n_aos'): {
                ('shell', 'n_shells'): {
                    ('prim?', 'n_prims'): {
                        'exp': {},
                        'coeff': {},
                    },
                    'molecule.atoms': {},
                }
            },
            'int1e': {
                'overlap',
                'elec_dipole',
            },
            'molecule': {
                ('atom', 'n_atoms'): {
                    'Z': {},
                    'coord': {},
                },
                'charge': {}
            },
        },
        'mo_space': {
            'n_mos': {},
            'n_elec': {},
            'multiplicity?': {},
        },
    },
    'n_transitions': {
        'n_states': {},
        'n_initial': {},
    },
    'n_mo_act': {
        'ci_space': {},
    },
    'n_configs': {
        'ci_space': {},
    },
}

parsing_dependency = {
    'multiplicity': 'fchk',
    'charge': 'fchk',
    'n_initial': {
        'route': {'log'},
    },
    'active_spaces': {
        'route': {'log'},
    }
}