from __future__ import annotations

import abc

from enum import unique, StrEnum
from typing import Any, TypeAlias, AnyStr, ClassVar

import attrs
import numpy as np

from mctools.core.resource import Resource

from ....core.error import ParsingError
from ....core.filehandler import FileWithPosition
from ....core.parser import Parser
from ....core.stepper import LineStepper
from ..route.route import IOps

__all__ = [
    'NewLinkParser',
    'MatrixParser',
    'MatrixLayout',
]


F: TypeAlias = FileWithPosition[AnyStr]
D = dict[Resource, Any]
LinkReturnType: TypeAlias = dict[Resource, Any]


@attrs.define(repr=True, eq=True)
class NewLinkParser(Parser, metaclass=abc.ABCMeta):
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


@unique
class MatrixLayout(StrEnum):
    FULL = 'Full'
    TRIL = 'Lower Triangular'


@attrs.mutable(repr=True, eq=True)
class MatrixParser:
    N_COLS_PER_BLOCK: ClassVar[int] = 5

    stepper: LineStepper = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )

    dtype: np.dtype.type = attrs.field(default=np.float32)

    def read_square_matrix(self) -> np.ndarray:
        n_rows, layout, first_block = self.read_block()
        if first_block is None:
            raise ParsingError('Could not read square matrix')

        shape = (n_rows, n_rows)

        matrix = np.zeros(shape, dtype=self.dtype)
        matrix[:, :self.N_COLS_PER_BLOCK] = first_block
        del first_block

        match layout:
            case MatrixLayout.FULL:
                self.read_full(matrix, col_offset=self.N_COLS_PER_BLOCK)
            case MatrixLayout.TRIL:
                self.read_tril_post(matrix, col_offset=self.N_COLS_PER_BLOCK)

        self.stepper.step_back()
        return matrix

    def read_block(self) -> tuple[int, MatrixLayout, list[list[float]]]:
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

        n_rows = len(array)
        if n_elem_read < n_rows * self.N_COLS_PER_BLOCK:
            layout = MatrixLayout.TRIL
        else:
            layout = MatrixLayout.FULL

        return n_rows, layout, array

    def read_tril_post(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
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

    def read_tril_exact(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
        n_rows, n_cols = matrix.shape
        for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
            self.stepper.step()
            for idx in range(jdx, n_rows):
                row = (self.stepper.readline()
                       .strip()
                       .replace('D', 'e')
                       .split())
                row = [matrix.dtype.type(v) for v in row[1:]]
                matrix[idx, jdx:jdx + len(row)] = row

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

# @attrs.define(repr=True)
# class LinkParser(Parser[LinkReturnType, AnyStr], metaclass=abc.ABCMeta):
#     START_ANCHOR_TEMPLATE: ClassVar[str] = '%s.exe'
#     END_ANCHOR: ClassVar[str] = 'Leave Link'
#
#     PARSABLE_RESOURCES: ClassVar[Resource] = Resource.NONE()
#     DEFAULT_LISTENERS: ClassVar[dict[Resource, Listener]] = {}
#
#     resources: Resource = attrs.field(
#         default=attrs.Factory(Resource.ALL),
#         validator=attrs.validators.instance_of(Resource),
#     )
#
#     iops: IOps = attrs.field(
#         default=attrs.Factory(dict),
#         validator=attrs.validators.deep_mapping(
#             key_validator=attrs.validators.ge(0),
#             value_validator=attrs.validators.instance_of(int),
#             mapping_validator=attrs.validators.instance_of(dict),
#         ),
#     )
#
#     resource_parsers: DispatchParser = attrs.field(init=False)
#
#     @resource_parsers.default
#     def _get_default_resource_parsers(self) -> DispatchParser:
#         return DispatchParser(terminator=self.END_ANCHOR)
#
#     stepper: LineStepper[AnyStr] = attrs.field(
#         factory=LineStepper,
#         validator=attrs.validators.instance_of(LineStepper),
#     )
#
#     def prepare(self, file: F, /) -> None:
#         super(LinkParser, self).prepare(file)
#         self.build_parsers()
#
#     def build_parsers(self) -> None:
#         for resource in self.resources:
#             if listener := self.DEFAULT_LISTENERS.get(resource):
#                 self.update_listener(listener)
#                 self.resource_parsers.add_listener(listener)
#
#     def update_listener(self, listener: Listener, /) -> None:
#         pass
#
#     def parse_file(self, fwp: F, /) -> tuple[D, F]:
#         self.stepper.take(fwp)
#
#         link_start_in = self.get_link_predicate()
#         if self.stepper.step_to(link_start_in, on_eof=self.on_parsing_error):
#             fwp = self.stepper.return_file()
#             return self.resource_parsers.parse(fwp)
#
#         fwp = self.stepper.return_file()
#         return {}, fwp
#
#     def get_link_predicate(self) -> Predicate[AnyStr]:
#         return self.stepper.get_anchor_predicate(self.get_start_anchor())
#
#     @classmethod
#     def get_start_anchor(cls) -> Anchor:
#         return cls.START_ANCHOR_TEMPLATE % cls.get_link().value
#
#     @classmethod
#     def get_link(cls) -> Link:
#         name = cls.__name__[:-len('Parser')]
#         return Link[name]


# @unique
# class MatrixLayout(StrEnum):
#     FULL = 'Full'
#     TRIL = 'Lower Triangular'
#
#
# @attrs.mutable(repr=True, eq=True)
# class MatrixParser:
#     N_COLS_PER_BLOCK: ClassVar[int] = 5
#
#     stepper: LineStepper = attrs.field(
#         factory=LineStepper,
#         validator=attrs.validators.instance_of(LineStepper),
#     )
#
#     dtype: np.dtype.type = attrs.field(default=np.float32)
#
#     def read_square_matrix(self) -> np.ndarray:
#         n_rows, layout, first_block = self.read_block()
#         if first_block is None:
#             raise ParsingError('Could not read square matrix')
#
#         shape = (n_rows, n_rows)
#
#         matrix = np.zeros(shape, dtype=self.dtype)
#         matrix[:, :self.N_COLS_PER_BLOCK] = first_block
#         del first_block
#
#         match layout:
#             case MatrixLayout.FULL:
#                 self.read_full(matrix, col_offset=self.N_COLS_PER_BLOCK)
#             case MatrixLayout.TRIL:
#                 self.read_tril_post(matrix, col_offset=self.N_COLS_PER_BLOCK)
#
#         self.stepper.step_back()
#         return matrix
#
#     def read_block(self) -> tuple[int, MatrixLayout, list[list[float]]]:
#         n_elem_read: int = 0
#         array: list[list[float]] = []
#         while '.' in (line := self.stepper.readline()):
#             row: list[float] = [
#                 float(v) for v in
#                 line.strip().replace('D', 'e').split()[1:]
#             ]
#             n_elem_read += len(row)
#
#             row.extend([0.] * (self.N_COLS_PER_BLOCK - len(row)))
#             array.append(row)
#
#         n_rows = len(array)
#         if n_elem_read < n_rows * self.N_COLS_PER_BLOCK:
#             layout = MatrixLayout.TRIL
#         else:
#             layout = MatrixLayout.FULL
#
#         return n_rows, layout, array
#
#     def read_tril_post(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
#         n_rows, n_cols = matrix.shape
#         for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
#             for idx in range(jdx, n_rows):
#                 row = (self.stepper.readline()
#                        .strip()
#                        .replace('D', 'e')
#                        .split())
#                 row = [matrix.dtype.type(v) for v in row[1:]]
#                 matrix[idx, jdx:jdx + len(row)] = row
#             self.stepper.step()
#
#         matrix += (matrix.T - np.diag(np.diag(matrix)))
#
#     def read_tril_exact(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
#         n_rows, n_cols = matrix.shape
#         for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
#             self.stepper.step()
#             for idx in range(jdx, n_rows):
#                 row = (self.stepper.readline()
#                        .strip()
#                        .replace('D', 'e')
#                        .split())
#                 row = [matrix.dtype.type(v) for v in row[1:]]
#                 matrix[idx, jdx:jdx + len(row)] = row
#
#     def read_full_exact(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
#         n_rows, n_cols = matrix.shape
#         for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
#             self.stepper.step()  # skip header
#             for idx in range(n_rows):
#                 row = (self.stepper.readline()
#                        .strip()
#                        .replace('D', 'e')
#                        .split())
#                 row = [matrix.dtype.type(v) for v in row[1:]]
#                 matrix[idx, jdx:jdx + len(row)] = row
#
#     def read_full(self, matrix: np.ndarray, /, col_offset: int = 0) -> None:
#         n_rows, n_cols = matrix.shape
#         for jdx in range(col_offset, n_cols, self.N_COLS_PER_BLOCK):
#             for idx in range(n_rows):
#                 row = (self.stepper.readline()
#                        .strip()
#                        .replace('D', 'e')
#                        .split())
#                 row = [matrix.dtype.type(v) for v in row[1:]]
#                 matrix[idx, jdx:jdx + len(row)] = row
#             self.stepper.step()
