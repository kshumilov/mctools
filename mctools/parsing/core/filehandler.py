from __future__ import annotations

import io
from typing import Generic, AnyStr, IO

import attrs


__all__ = [
    'FileWithPosition',
    'FileHandler',

    'DoesNotHaveAFile',
    'AlreadyHasFile',
]


@attrs.define(repr=True, eq=True)
class FileWithPosition(Generic[AnyStr]):
    file: IO[AnyStr] = attrs.field(
        validator=attrs.validators.instance_of(io.IOBase)
    )
    n_lines_read: int = attrs.field(
        default=0,
        converter=int,
        validator=attrs.validators.ge(0),
    )
    last_line: AnyStr = attrs.field()

    @last_line.default
    def _get_last_line(self) -> str | bytes:
        if self.is_binary:
            return b''
        return ''

    @file.validator  # type: ignore
    def _validate_file(self, attribute: attrs.Attribute, value: IO[AnyStr]) -> None:
        if value.closed:
            raise ValueError('File must be open, readable, and seekable')

    def forward(self) -> bool:
        self.n_lines_read += 1
        self.last_line = self.file.readline()
        return bool(self.last_line)

    def backward(self) -> bool:
        old_offset = self.file.tell()
        new_offset = old_offset - len(self.last_line)
        if new_offset != old_offset:
            self.n_lines_read -= 1
            self.file.seek(new_offset)
            self.last_line = type(self.last_line)()
        return new_offset != 0

    @property
    def is_binary(self) -> bool:
        return 'b' in self.file.mode

    # @property
    # def is_parsable(self) -> bool:
    #     try:
    #         attrs.validate(self)  # mypy: ignore
    #         return True
    #     except ValueError:
    #         return False


class FileHandlingError(Exception):
    pass


class AlreadyHasFile(FileHandlingError):
    pass


class DoesNotHaveAFile(FileHandlingError):
    pass


@attrs.define(repr=True, eq=True)
class FileHandler(Generic[AnyStr]):
    fwp: FileWithPosition[AnyStr] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(FileWithPosition)
        ),
    )

    @classmethod
    def from_file(cls, file: IO[AnyStr]) -> FileHandler[AnyStr]:
        return cls(FileWithPosition(file))

    def take(self, fwp: FileWithPosition[AnyStr], /) -> None:
        if self.fwp is not None:
            raise AlreadyHasFile()

        attrs.validate(fwp)  # type: ignore
        self.fwp = fwp

    def take_from(self, other: FileHandler[AnyStr], /) -> None:
        if self.fwp is not None:
            raise AlreadyHasFile()

        fwp = other.return_file()
        attrs.validate(fwp)  # type: ignore
        self.fwp = fwp

    def return_file(self, /) -> FileWithPosition[AnyStr]:
        if self.fwp is None:
            raise DoesNotHaveAFile()

        fwp, self.fwp = self.fwp, None
        return fwp
