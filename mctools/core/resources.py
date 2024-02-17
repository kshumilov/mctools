from __future__ import annotations

from enum import IntFlag, auto

__all__ = [
    'Resources',
]


class Resources(IntFlag):
    NONE = 0

    # Geometry & Molecular information
    atcoords = auto()
    atnums = auto()
    charge = auto()
    multiplicity = auto()

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
    energy = auto()
    ci_vecs = auto()
    one_rdms = auto()
    one_tdms = auto()
    osc_dipole = auto()

    ao_int1e_elec_multi = ao_int1e_elec_dipole | ao_int1e_elec_quad | ao_int1e_elec_oct | ao_int1e_elec_hex
    ao_int1e_stv = ao_int1e_overlap | ao_int1e_kinetic | ao_int1e_potential
