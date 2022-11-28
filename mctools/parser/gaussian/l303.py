import re

multipole_header = re.compile(r'Multipole matrices ')  # Appears 3 times
fermi_header = re.compile(r'Fermi contact integrals:')  # Not square

