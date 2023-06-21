from __future__ import annotations

import numpy as np
import pandas as pd

from scipy import constants

__all__ = [
    'g_e', 'alpha',

    'Eh2eV', 'eV2Eh', 'wn2eV',
    'sigma_0', 'sigma_x', 'sigma_y', 'sigma_z', 'sigma_vector',

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
PeriodicTable = pd.read_csv('periodic_table.csv')

sigma_0 = np.eye(2)
sigma_z = np.asarray([[1, 0], [0, -1]])
sigma_x = np.asarray([[0, 1], [1,  0]])
sigma_y = 1j * np.asarray([[0, -1], [1,  0]])
sigma_vector = np.stack([sigma_x, sigma_y, sigma_z])

ANGULAR_MOMENTUM_SYMBS = np.asarray(['S', 'P', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
