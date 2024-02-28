from __future__ import annotations

import os

import numpy as np
import pandas as pd

from scipy import constants

__all__ = [
    'g_e', 'alpha',

    'Eh2eV', 'eV2Eh', 'wn2eV',
    'I2', 'SX', 'SY', 'SZ', 'SIGMA_VECTOR', 'PAULI_VECTOR',

    'PeriodicTable',

    'Bh2Ang',

    'ANGULAR_MOMENTUM_SYMBS',
]

alpha = constants.alpha
g_e = constants.value('electron g factor')

Eh2eV = constants.value('Hartree energy in eV')
eV2Eh = 1 / Eh2eV
wn2eV = 1.23981e-4

Bh2Ang = 0.529177249
PeriodicTable = pd.read_csv(os.path.join(os.path.dirname(__file__), 'periodic_table.csv')).set_index('AtomicNumber')

I2 = np.eye(2)
SZ = np.asarray([[1, 0], [0, -1]])
SX = np.asarray([[0, 1], [1, 0]])
SY = 1j * np.asarray([[0, -1], [1, 0]])
SIGMA_VECTOR = np.stack([SX, SY, SZ])
PAULI_VECTOR = np.stack([I2, SX, SY, SZ])

ANGULAR_MOMENTUM_SYMBS = np.asarray(['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
