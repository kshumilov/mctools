from __future__ import annotations

from enum import Flag, auto, unique
from functools import reduce


__all__ = [
    'Resource',
]


@unique
class Resource(Flag):
    # Geometry & Molecular information
    mol_atcoords = auto()
    mol_atnums = auto()
    mol_charge = auto()
    mol_multiplicity = auto()

    # Orbital basis
    ao_basis = auto()

    # Integrals
    ao_int1e_overlap = auto()
    ao_int1e_kinetic = auto()
    ao_int1e_potential = auto()

    ao_int1e_elec_dipole = auto()
    ao_int1e_elec_quad = auto()
    ao_int1e_elec_oct = auto()
    ao_int1e_elec_hex = auto()

    # Multiconfigurational Wavefunction information
    ci_energy = auto()
    ci_nconfigs = auto()
    ci_nstates = auto()
    ci_vecs = auto()
    ci_space = auto()
    ci_saweights = auto()
    ci_int1e_rdms = auto()
    ci_int1e_tdms = auto()
    ci_osc = auto()
    ci_spin = auto()

    @classmethod
    def NONE(cls) -> Resource:
        return cls(0)

    @classmethod
    def ALL(cls) -> Resource:
        return cls(reduce(cls.__or__, list(cls), cls(0)))

    @classmethod
    def STV(cls) -> Resource:
        return cls(cls.ao_int1e_overlap | cls.ao_int1e_kinetic | cls.ao_int1e_potential)

    @classmethod
    def CI(cls) -> Resource:
        return cls(cls.ci_energy | cls.ci_vecs | cls.ci_space |
                   cls.ci_int1e_rdms | cls.ci_int1e_tdms | cls.ci_osc |
                   cls.ci_saweights | cls.ci_spin)

    # @classmethod
    # def ci_states(cls) -> set[Resource]:
    #     return {cls.ci_energy, cls.ci_vecs}
    #
    # @classmethod
    # def ao_int1e_elec_multi(cls):
    #     return {cls.ao_int1e_elec_dipole, cls.ao_int1e_elec_quad,
    #             cls.ao_int1e_elec_oct, cls.ao_int1e_elec_hex}
