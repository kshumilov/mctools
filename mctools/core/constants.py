from __future__ import annotations

import numpy as np
from scipy import constants

__all__ = [
    'Eh2eV',
    'eV2Eh',
    'wn2eV',

    'ANGULAR_MOMENTUM_SYMBS',
]


Eh2eV = constants.value('Hartree energy in eV')
eV2Eh = 1 / Eh2eV
wn2eV = 1.23981e-4

ANGULAR_MOMENTUM_SYMBS = np.asarray(['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
