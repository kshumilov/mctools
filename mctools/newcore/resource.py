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
    mol_nelec = auto()
    mol_multiplicity = auto()

    # Atomic Orbital basis
    ao_basis = auto()
    ao_basis_shell = auto()
    ao_basis_atom = auto()
    ao_basis_l = auto()
    ao_basis_ml = auto()

    ao_basis_prims_coef = auto()
    ao_basis_prims_exp = auto()

    ao_basis_shells_coords = auto()
    ao_basis_shells_size = auto()
    ao_basis_shells_atom = auto()
    ao_basis_shells_ang = auto()

    # MO Basis
    mo_basis = auto()
    mo_basis_molorb = auto()
    mo_basis_ansatz = auto()

    # Integrals
    ao_int1e_overlap = auto()
    ao_int1e_kinetic = auto()
    ao_int1e_potential = auto()

    ao_int1e_elec_dipole = auto()
    ao_int1e_elec_quad = auto()
    ao_int1e_elec_oct = auto()
    ao_int1e_elec_hex = auto()

    # Multiconfigurational Wavefunction information
    ci_states = auto()
    ci_state_idx = auto()
    ci_energies = auto()
    ci_vecs = auto()
    ci_graph = auto()
    ci_sa_weights = auto()
    ci_int1e_rdms = auto()
    ci_spin = auto()

    ci_initial_idx = auto()
    ci_final_idx = auto()
    ci_int1e_tdms = auto()
    ci_osc = auto()

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
        return cls(cls.ci_energies | cls.ci_vecs | cls.ci_graph |
                   cls.ci_int1e_rdms | cls.ci_int1e_tdms | cls.ci_osc |
                   cls.ci_sa_weights | cls.ci_spin)

    # @classmethod
    # def ci_states(cls) -> set[Resource]:
    #     return {cls.ci_energy, cls.ci_vecs}
    #
    # @classmethod
    # def ao_int1e_elec_multi(cls):
    #     return {cls.ao_int1e_elec_dipole, cls.ao_int1e_elec_quad,
    #             cls.ao_int1e_elec_oct, cls.ao_int1e_elec_hex}
