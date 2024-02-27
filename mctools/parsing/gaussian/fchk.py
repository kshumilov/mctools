from __future__ import annotations

from typing import AnyStr, TypeAlias, ClassVar

import attrs
import numpy as np
from numpy import ndarray, dtype

from mctools.cli.console import console
from mctools.core.resource import Resource

from ..core.error import AnchorNotFound
from ..core.parser import Parser, FWP
from ..core.pattern import ProcessedPattern, simple_int_tmplt, simple_float_tmplt
from ..core.stepper import LineStepper

__all__ = [
    'FchkParser',
    # 'mo_to_fchk_str',
]


GAUSSIAN_SHELL_TYPES: dict[int, str] = {
    0: 'S',
    -1: 'SP', 1: 'P',
    -2: 'D',  2: 'd',
    -3: 'F',  3: 'f',
    -4: 'G',  4: 'g',
    -5: 'H',  5: 'h',
}

GAUSSIAN_ML_AO: dict[str, list[int]] = {
    'S': [0],
    'P': [1, -1, 0],
    'D': [0, 1, -1, 2, -2],
    'F': [0, 1, -1, 2, -2, 3, -3],
    'G': [0, 1, -1, 2, -2, 3, -3, 4, -4],
    'H': [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5],
}

ScalarType: TypeAlias = np.float_ | np.int_ | np.str_


def process_data_type(char: str, /) -> ScalarType:
    match char:
        case 'R':
            return np.float_
        case 'I':
            return np.int_
        case 'C':
            return np.str_
        case _:
            raise ValueError(f'Invalid data_type: {char}')


@attrs.define(eq=True, repr=True)
class FchkParser(Parser):
    DEFAULT_HEADERS = [
        'Charge',
        'Multiplicity',
        'Number of electrons',
        'Atomic Numbers',
        'Current cartesian coordinates',
        'Shell types',
        'Number of primitives per shell',
        'Shell to atom map',
        'Primitive exponents',
        'Contraction coefficients',
        'Coordinates of each shell',
        'Alpha MO coefficients',
        'Beta MO coefficients',
    ]

    MethodPatt: ClassVar[ProcessedPattern] = ProcessedPattern(
        r'(?P<calc>[a-zA-Z]+)\s*'
        r'(?P<method>(?P<mo_anzats>R|RO|U|G)\w+)'
        r'\s*(?P<basis_name>\w+)'
    )

    FCHK_HEADER_PATT: ClassVar[ProcessedPattern] = ProcessedPattern(
        r'(?P<header>([A-Z][a-zA-Z]+)(\s[a-zA-Z]+)+)\s+'
        r'(?P<dtype>[IRCHL])\s+'
        r'((N=\s+(?P<size>\d+))|(?P<value>[+-]?\d+(\.\d+[dDeE]\+\d+)?))',
        constructor='header',
        group_maps={
            'size': lambda s: int(s) if s else s,
            'dtype': process_data_type,
        },
        match_funcs={
            'value': lambda m: int(m.pop('value')) if m['size'] else m['dtype'](m.pop('value'))
        }
    )

    requested: Resource = attrs.field(
        factory=Resource.ALL,
        validator=attrs.validators.instance_of(Resource),
    )

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    FCHK_SCALAR_PATT = ProcessedPattern(
        r'%s\s*(?P<dt>I|R)\s*' +
        (r'(?P<v>%s|%s)' % (simple_int_tmplt, simple_float_tmplt)),
        constructor=lambda dt, v: dt(v), group_maps={'dt': process_data_type}
    )

    FCHK_ARRAY_PATT = ProcessedPattern(
        r'%s\s*(?P<dtype>I|R)\s*N=\s*(?P<size>\d+)',
        'array_info',
        group_maps={'size': int, 'dtype': process_data_type}
    )

    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[dict[Resource, np.ndarray], FWP[AnyStr]]:
        console.rule(f'Fchk file: {fwp.file.name}')

        self.stepper.take(fwp)
        result = {}

        info = self.read_description()
        mo_anzats = info['mo_anzats']
        result.update({Resource.mo_basis_ansatz: np.asarray(mo_anzats, dtype='S')})

        charge = self.read_scalar('Charge')
        result.update({Resource.mol_charge: np.asarray(charge)})

        multiplicity = self.read_scalar('Multiplicity')
        result.update({Resource.mol_charge: np.asarray(multiplicity)})

        n_elec = self.read_scalar('Number of electrons')
        result.update({Resource.mol_nelec: np.asarray(n_elec)})

        n_ao = self.read_scalar('Number of basis functions')

        result.update(self.read_molecular_geometry())
        result.update(self.read_basis())

        result.update(self.read_molecular_orbitals(n_ao, mo_anzats))

        console.print('Finished parsing FCHK file')
        return result, self.stepper.return_file()

    def read_description(self) -> dict[str, str]:
        console.print('Parsing FCHK description...')

        short_title = self.stepper.readline()

        if match := self.MethodPatt.search(self.stepper.readline()):
            return {'short_title': short_title.strip(), **match}

        raise AnchorNotFound('Could not find FCHK method line')

    def read_molecular_geometry(self) -> dict[Resource, np.ndarray]:
        console.print('Parsing Molecule geometry...')

        atomic_number = self.read_array('Atomic numbers')
        coords = self.read_array('Current cartesian coordinates')
        coords = coords.reshape((-1, 3))

        return {
            Resource.mol_atnums: atomic_number,
            Resource.mol_atcoords: coords,
        }

    def read_basis(self) -> dict[Resource, np.ndarray]:
        console.print('Parsing Atomic Orbital Basis...')

        # Shell information (Part 1)
        shell_code = self.read_array('Shell types')
        shell_size = self.read_array('Number of primitives per shell')
        shell2atom = self.read_array('Shell to atom map')
        shell2atom -= 1  # Start atom enumeration from 0

        # Primitive information (Part 2)
        primitive_exponents = self.read_array('Primitive exponents')
        contraction_coeffs = self.read_array('Contraction coefficients')
        primitives = {
            Resource.ao_prim_coef: contraction_coeffs,
            Resource.ao_prim_exp: primitive_exponents,
        }

        shell_coords = self.read_array('Coordinates of each shell')
        shell_coords = shell_coords.reshape((-1, 3))

        shells = {
            Resource.ao_shell_atom: shell2atom,
            Resource.ao_shell_size: shell_size,
            Resource.ao_shell_l: np.asarray([GAUSSIAN_SHELL_TYPES[c] for c in shell_code], dtype='S'),
            Resource.ao_shell_coords: shell_coords,
        }

        return shells | primitives

    def read_molecular_orbitals(self, n_ao: int, mo_anzats: str) -> dict[Resource, np.ndarray]:
        console.print('Parsing Molecular Orbital Basis...')

        molorb_raw_a = self.read_array('Alpha MO coefficients')

        match mo_anzats:
            case 'R':
                molorb = molorb_raw_a.reshape(n_ao, n_ao)
            case 'U':
                molorb_raw_b = self.read_array('Beta MO coefficients')
                molorb = np.zeros((n_ao * 2, n_ao), dtype=np.float_)
                molorb[:n_ao] = molorb_raw_a.reshape(n_ao, n_ao)
                molorb[n_ao:] = molorb_raw_b.reshape(n_ao, n_ao)
            case 'G':
                molorb = molorb_raw_a[0::2] + molorb_raw_a[1::2] * 1.j
                molorb = molorb.reshape(-1, n_ao * 2)  # (#MOs, #AOs)
            case _:
                raise ValueError(f'Invalid MO restriction '
                                 f'is specified: {mo_anzats}')

        return {Resource.mo_basis_molorb: molorb}

    def read_scalar(self, header: str) -> int | float | str:
        anchor_in = self.stepper.get_anchor_predicate(header)
        self.stepper.step_to(anchor_in, check_last_read=True, on_eof='raise')
        patt = self.FCHK_SCALAR_PATT.update_pattern(header)
        return patt.match(self.stepper.fwp.last_line)

    def read_array(self, header: str) -> ndarray[ScalarType, dtype[ScalarType]]:
        anchor_in = self.stepper.get_anchor_predicate(header)
        self.stepper.step_to(anchor_in, check_last_read=True, on_eof='raise')
        patt = self.FCHK_ARRAY_PATT.update_pattern(header)
        arr_info = patt.match(self.stepper.fwp.last_line)

        arr = []
        while self.stepper.step() and len(arr) < arr_info.size:
            arr.extend(self.stepper.fwp.last_line.split())

        return np.asarray(arr, dtype=arr_info.dtype)


# class FchkParser(FileParser):
#     DefaultStepper = FchkStepper
#
#     DEFAULT_HEADERS = [
#         'Charge',
#         'Multiplicity',
#         'Number of electrons',
#         'Atomic Numbers',
#         'Current cartesian coordinates',
#         'Shell types',
#         'Number of primitives per shell',
#         'Shell to atom map',
#         'Primitive exponents',
#         'Contraction coefficients',
#         'Coordinates of each shell',
#         'Alpha MO coefficients',
#         'Beta MO coefficients',
#     ]
#
#     def __init__(self, *args, headers: Literal['all', 'default'] | list = 'default', stepper=None, **kwargs) -> None:
#         super(FileParser, self).__init__(*args, stepper=None, **kwargs)
#         self.headers = headers
#
#     def read_description(self) -> dict[str, str]:
#         short_title = self.stepper.readline()
#
#         if match := self.stepper.MethodPatt.search(self.stepper.readline()):
#             return {'short_title': short_title.strip(), **match}
#
#         self.stepper.target_not_found(message='Could not find FCHK method line')
#
#     def read_molecular_geometry(self) -> pd.DataFrame:
#         atomic_number = self.stepper.read_array('Atomic numbers')
#         coords = self.stepper.read_array('Current cartesian coordinates')
#         coords = coords.reshape((-1, 3))
#
#         return pd.DataFrame({
#             Molecule.ATOMIC_NUMBER_COL: atomic_number,
#             Molecule.ATOM_COL: np.arange(atomic_number.shape[0]) + 1,
#             **{col: arr for col, arr in zip(Molecule.COORDS_COLS, coords.T)}
#         })
#
#     def read_basis(self) -> dict[str, pd.DataFrame]:
#         # Shell information (Part 1)
#         shell_code = self.stepper.read_array('Shell types')
#         shell_size = self.stepper.read_array('Number of primitives per shell')
#         shell2atom = self.stepper.read_array('Shell to atom map')
#         shell2atom -= 1  # Start atom enumeration from 0
#
#         # Primitive information (Part 2)
#         primitive_exponents = self.stepper.read_array('Primitive exponents')
#         contraction_coeffs = self.stepper.read_array('Contraction coefficients')
#         primitives = pd.DataFrame({
#             'C': contraction_coeffs,
#             'alpha': primitive_exponents,
#         })
#
#         shell_coords = self.stepper.read_array('Coordinates of each shell')
#         shell_coords = shell_coords.reshape((len(shell_code), 3))
#
#         shells = pd.DataFrame({
#             'code': shell_code,
#             'n_prim': shell_size,
#             'atom': shell2atom,
#         })
#         shells['type'] = shells['code'].map(GAUSSIAN_SHELL_TYPES)
#         shells[['x', 'y', 'z']] = shell_coords
#
#         primitives['shell'] = np.repeat(shells.index, shells.n_prim)
#
#         # Atomic Orbitals information
#         aos = defaultdict(list)
#         for shell_idx, shell in shells.iterrows():
#             for shell_part in shell.type:
#                 mls = GAUSSIAN_ML_AO[shell_part.capitalize()]
#                 l = np.max(mls)
#                 n_ao_per_shell = len(mls)
#
#                 aos['shell'].extend([shell_idx] * n_ao_per_shell)
#                 aos['atom'].extend([shell.atom] * n_ao_per_shell)
#                 aos['l'].extend([l] * n_ao_per_shell)
#                 aos['ml'].extend(mls)
#
#         aos = pd.DataFrame(aos)
#         #
#         # if atomic_number is not None:
#         #     atomic_number = np.asarray(atomic_number)
#         #     shells['element'] = atomic_number[shells['atom']]
#         #     aos['element'] = atomic_number[aos['atom']]
#
#         return {
#             'shells': shells,
#             'aos': aos,
#             'prims': primitives,
#         }
#
#     def read_molecular_orbitals(self, n_ao: int, mo_anzats: str) -> np.ndarray:
#         molorb_raw_a = self.stepper.read_array('Alpha MO coefficients')
#
#         match mo_anzats:
#             case 'R':
#                 molorb = molorb_raw_a.reshape(n_ao, n_ao)
#             case 'U':
#                 molorb_raw_b = self.stepper.read_array('Beta MO coefficients')
#                 molorb = np.zeros((n_ao * 2, n_ao), dtype=np.float_)
#                 molorb[:n_ao] = molorb_raw_a.reshape(n_ao, n_ao)
#                 molorb[n_ao:] = molorb_raw_b.reshape(n_ao, n_ao)
#             case 'G':
#                 molorb = molorb_raw_a[0::2] + molorb_raw_a[1::2] * 1.j
#                 molorb = molorb.reshape(-1, n_ao * 2)  # (#MOs, #AOs)
#             case _:
#                 raise ValueError(f'Invalid MO restriction '
#                                  f'is specified: {mo_anzats}')
#
#         return molorb
#
#     def parse_file(self, state: IO[AnyStr]) -> Any:
#         header = self.read_description()
#
#     def assign_task(self, option: FchkParserOptions) -> Optional[callable | tuple[callable, str]]:
#         match option:
#             case self.options_cls.header:
#                 return self.read_description
#             case self.options_cls.mol_charge:
#                 return partial(self.stepper.read_scalar, 'Charge'), f'{self.stepper.read_scalar.__name__}'
#             case self.options_cls.mol_multiplicity:
#                 return partial(self.stepper.read_scalar, 'Multiplicity'), f'{self.stepper.read_scalar.__name__}'
#             case self.options_cls.n_elec_a:
#                 return partial(self.stepper.read_scalar, 'Number of alpha electrons'), f'{self.stepper.read_scalar.__name__}'
#             case self.options_cls.n_elec_b:
#                 return partial(self.stepper.read_scalar, 'Number of beta electrons'), f'{self.stepper.read_scalar.__name__}'
#             case self.options_cls.n_atomic_orbitals:
#                 return partial(self.stepper.read_scalar, 'Number of basis functions'), f'{self.stepper.read_scalar.__name__}'
#             case self.options_cls.molecular_geometry:
#                 return self.read_molecular_geometry, f'{self.read_molecular_geometry.__name__}'
#             case self.options_cls.basis:
#                 return self.read_basis, f'{self.read_basis.__name__}'
#             case self.options_cls.molecular_orbitals:
#                 try:
#                     n_ao = self.data[self.options_cls.n_atomic_orbitals]
#                     mo_anzats = self.data[self.options_cls.header]['mo_anzats']
#                 except KeyError as exc:
#                     raise RequirementNotSatisfied from exc
#
#                 return partial(self.read_molecular_orbitals, n_ao, mo_anzats), f'{self.read_molecular_orbitals.__name__}'
#             case _:
#                 return None
#
#     def process(self: 'OptionedFileParser', result: Data) -> Data:
#         processed_result = {}
#         for option in self.options_cls:
#             if option in result:
#                 data = result.pop(option)
#                 match option:
#                     case self.options_cls.molecular_geometry:
#                         charge = processed_result.pop(self.options_cls.mol_charge)
#                         multiplicity = processed_result.pop(self.options_cls.mol_multiplicity)
#                         processed_result[option] = Molecule(
#                             data,
#                             charge=charge, multiplicity=multiplicity,
#                             source=self.stepper_class.fwp.parser_class,
#                         )
#                     case self.options_cls.basis:
#                         molecule = processed_result.pop(self.options_cls.molecular_geometry)
#                         mo_anzats = processed_result.pop('mo_anzats')
#                         processed_result[option] = {
#                             'basis': Basis.from_dict(
#                                 data, df_key='aos',
#                                 restriction=mo_anzats, molecule=molecule
#                             )
#                         }
#                         processed_result[option].update(data)
#                     case _:
#                         if isinstance(data, dict):
#                             processed_result.update(data)
#                         else:
#                             processed_result[option] = data
#         processed_result.update(result)
#         return processed_result


# def mo_to_fchk_str(mo: np.ndarray, /, n_per_line: int = 5, fmt: str = '%16.8E') -> str:
#     values = mo.flatten()
#
#     if values.dtype is np.dtype(np.complex128):
#         new_values = np.zeros(len(values) * 2, dtype=np.float_)
#         new_values[0::2] = values.real
#         new_values[1::2] = values.imag
#         values = new_values
#
#     strs = [fmt % v for v in values]
#
#     n_batches = len(strs) // n_per_line
#
#     s = ''
#     for i in range(n_batches):
#         s += ''.join(strs[i * n_per_line:(i + 1) * n_per_line]) + '\n'
#
#     return s


# def parse_gdvfchk(filename: str, read_funcs: list[Callable], /, **kwargs) -> ParsingResultType:
#     result: ParsingResultType = {'source': filename, **kwargs}
#
#     with open(filename, 'r') as file:
#         print(f'Reading {filename}')
#         result, line = parse_file(file, read_funcs, result)
#
#     return result
