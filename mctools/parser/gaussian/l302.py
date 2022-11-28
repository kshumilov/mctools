import re


ovlp_header = re.compile(r'\*\*\* Overlap \*\*\*')
hcore_header = re.compile(r'\*\*\*\*\*\* Core Hamiltonian \*\*\*\*\*\*')
veffp_header = re.compile(r'\*\*\* Veff \(p space\) \*\*\*')
trelr_header = re.compile(r'\*\*\* Trel \(r space\) \*\*\*')
veffr_header = re.compile(r'\*\*\* Veff \(r space\) \*\*\*')
so_header = re.compile(r'\*\*\* SO unc. \*\*\*')  # Appears 3 times
x2c_header = re.compile(r'DK / X2C integrals')  # Appears 5 times
ortho_header = re.compile(r'Orthogonalized basis functions:')
