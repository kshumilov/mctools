from __future__ import annotations

import abc
import warnings
from enum import unique, StrEnum
from typing import Any, TypeAlias, AnyStr, ClassVar

import attrs
import numpy as np
from icecream import ic

from core.resource import Resource

from parsing.core.error import EOFReached, ParsingError
from parsing.core.filehandler import FileWithPosition
from parsing.core.parser import Parser, Listener, DispatchParser
from parsing.core.stepper import LineStepper, Predicate, Anchor
from parsing.gaussian.log.route.route import Link, IOps

__all__ = [
    'LinkParser',
    'RealMatrixParser',
    'MatrixSymmetry',
]


F: TypeAlias = FileWithPosition[AnyStr]
D = dict[Resource, Any]
LinkReturnType: TypeAlias = dict[Resource, Any]


@attrs.define(repr=True)
class LinkParser(Parser[LinkReturnType, AnyStr], metaclass=abc.ABCMeta):
    START_ANCHOR_TEMPLATE: ClassVar[str] = '%s.exe'
    END_ANCHOR: ClassVar[str] = 'Leave Link'

    PARSABLE_RESOURCES: ClassVar[Resource] = Resource.NONE()
    DEFAULT_LISTENERS: ClassVar[dict[Resource, Listener]] = {}

    resources: Resource = attrs.field(
        default=attrs.Factory(Resource.ALL),
        validator=attrs.validators.instance_of(Resource),
    )

    iops: IOps = attrs.field(
        default=attrs.Factory(dict),
        validator=attrs.validators.deep_mapping(
            key_validator=attrs.validators.ge(0),
            value_validator=attrs.validators.instance_of(int),
            mapping_validator=attrs.validators.instance_of(dict),
        ),
    )

    resource_parsers: DispatchParser = attrs.field(init=False)

    @resource_parsers.default
    def _get_default_resource_parsers(self) -> DispatchParser:
        return DispatchParser(terminator=self.END_ANCHOR)

    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    def prepare(self, file: F, /) -> None:
        super(LinkParser, self).prepare(file)
        self.build_parsers()

    def build_parsers(self) -> None:
        for resource in self.resources:
            if listener := self.DEFAULT_LISTENERS.get(resource):
                self.update_listener(listener)
                self.resource_parsers.add_listener(listener)

    def update_listener(self, listener: Listener, /) -> None:
        pass

    def parse_file(self, fwp: F, /) -> tuple[D, F]:
        self.stepper.take(fwp)

        link_start_in = self.get_link_predicate()
        if self.stepper.step_to(link_start_in, on_eof=self.on_parsing_error):
            fwp = self.stepper.return_file()
            return self.resource_parsers.parse(fwp)

        fwp = self.stepper.return_file()
        return {}, fwp

    def get_link_predicate(self) -> Predicate[AnyStr]:
        return self.stepper.get_anchor_predicate(self.get_start_anchor())

    @classmethod
    def get_start_anchor(cls) -> Anchor:
        return cls.START_ANCHOR_TEMPLATE % cls.get_link().value

    @classmethod
    def get_link(cls) -> Link:
        name = cls.__name__[:-len('Parser')]
        return Link[name]


@unique
class MatrixSymmetry(StrEnum):
    FULL = 'full'
    TRIL = 'lower_triangular'
    TRIU = 'upper_triangular'
    DIAG = 'diagonal'
    HERM = 'hermitian'


@attrs.mutable(repr=True, eq=True)
class RealMatrixParser(Parser):
    N_COLS_PER_BLOCK: ClassVar[int] = 5

    stepper: LineStepper = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    n_rows: int | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(int),
        validator=attrs.validators.optional(attrs.validators.ge(0))
    )

    n_cols: int | None = attrs.field(
        default=None,
        converter=attrs.converters.optional(int),
        validator=attrs.validators.optional(attrs.validators.ge(0))
    )

    symmetry: MatrixSymmetry = attrs.field(
        default=MatrixSymmetry.FULL,
        validator=attrs.validators.instance_of(MatrixSymmetry)
    )

    dtype = attrs.field(default=np.float32)

    @property
    def is_square(self) -> bool:
        return self.n_rows == self.n_cols

    @property
    def is_shape_known(self) -> bool:
        return not (self.n_rows is None or self.n_cols is None)

    def parse_file(self, fwp: F, /) -> tuple[np.ndarray, F]:
        self.stepper.take(fwp)
        matrix = self.read_square_matrix()
        return matrix, self.stepper.return_file()

    def read_matrix(self, matrix: np.ndarray | None = None) -> np.ndarray:
        if self.is_shape_known and self.symmetry == MatrixSymmetry.FULL:
            if matrix is None:
                matrix = np.zeros((self.n_rows, self.n_cols), dtype=self.dtype)

            # self.stepper.readline()  # Skip header
            self.read_full_exact(matrix, col_offset=0)
            return matrix
        return self.read_square_matrix()

    def read_square_matrix(self) -> np.ndarray:
        first_block = self.read_block()
        if first_block is None:
            raise ParsingError('Could not read square matrix')

        self.n_cols = self.n_rows

        matrix = np.zeros((self.n_rows, self.n_cols), dtype=self.dtype)
        matrix[:, :self.N_COLS_PER_BLOCK] = first_block
        del first_block

        match self.symmetry:
            case MatrixSymmetry.FULL:
                self.read_full(matrix, col_offset=self.N_COLS_PER_BLOCK)
            case MatrixSymmetry.TRIL:
                self.read_tril(matrix, col_offset=self.N_COLS_PER_BLOCK)

        self.stepper.step_back()
        return matrix

    def read_block(self) -> list[list[float]] | None:
        if not self.stepper.step():  # Skip Header
            match self.on_parsing_error:
                case 'raise':
                    raise EOFReached()
                case 'warn':
                    warnings.warn('EOF Reached')
                    return None
                case 'skip':
                    return None

        n_elem_read: int = 0
        array: list[list[float]] = []
        while '.' in (line := self.stepper.readline()):
            row: list[float] = [
                float(v) for v in
                line.strip().replace('D', 'e').split()[1:]
            ]
            n_elem_read += len(row)

            row.extend([0.] * (self.N_COLS_PER_BLOCK - len(row)))
            array.append(row)

        self.n_rows = len(array)
        if n_elem_read < self.n_rows * self.N_COLS_PER_BLOCK:
            self.symmetry = MatrixSymmetry.TRIL
        else:
            self.symmetry = MatrixSymmetry.FULL

        return array

    def read_tril(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
        n_rows, n_cols = matrix.shape
        for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
            for idx in range(jdx, n_rows):
                row = (self.stepper.readline()
                       .strip()
                       .replace('D', 'e')
                       .split())
                row = [matrix.dtype.type(v) for v in row[1:]]
                matrix[idx, jdx:jdx + len(row)] = row
            self.stepper.step()

        matrix += (matrix.T - np.diag(np.diag(matrix)))

    def read_full_exact(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
        n_rows, n_cols = matrix.shape
        for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
            self.stepper.step()  # skip header
            for idx in range(n_rows):
                row = (self.stepper.readline()
                       .strip()
                       .replace('D', 'e')
                       .split())
                row = [matrix.dtype.type(v) for v in row[1:]]
                matrix[idx, jdx:jdx + len(row)] = row

    def read_full(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
        n_rows, n_cols = matrix.shape
        for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
            for idx in range(n_rows):
                row = (self.stepper.readline()
                       .strip()
                       .replace('D', 'e')
                       .split())
                row = [matrix.dtype.type(v) for v in row[1:]]
                matrix[idx, jdx:jdx + len(row)] = row
            self.stepper.step()
