from __future__ import annotations

from typing import IO, AnyStr, Callable, Any, Iterator, TypeVar, Generic, Literal, TypeAlias

import attr

from .error import ParsingError, EOFReached, InvalidFile
from .utils import check_file, short_file_repr

__all__ = [
    # Classes
    'LineStepper',
    'FileStepper',

    # Types
    'LineStepperType',
    'Predicate',
    'Anchor',
]


Processor = Callable[[AnyStr], bool | Any]


@attr.define(repr=True)
class FileHandler(Generic[AnyStr]):
    file: IO[AnyStr] | None = attr.field(default=None, init=False, repr=short_file_repr)

    def set_file(self, file: IO[AnyStr] | None, /, *, check: bool = True) -> None:
        if check and not check_file(file):
            raise ValueError(f'File is not workable: {file!r}')

        self.file = file

    def unset_file(self) -> IO[AnyStr]:
        file = self.file
        self.file = None
        return file

    @property
    def workable(self) -> bool:
        return check_file(self.file)

    @classmethod
    def from_file(cls, file: IO[AnyStr]) -> 'FileHandler':
        handler = cls()
        handler.set_file(file, check=True)
        return handler


@attr.define(repr=False)
class LineStepper(FileHandler[AnyStr]):
    on_eof: Literal['raise', 'skip'] = attr.field(
        default='skip',
        validator=attr.validators.instance_of(str)
    )
    line: AnyStr | None = attr.field(default=None, init=False)

    def set_file(self, file: IO[AnyStr] | None, /, *, check: bool = True, line: AnyStr = None) -> None:
        super(LineStepper, self).set_file(file, check=check)
        self.line = line

    def unset_file(self) -> tuple[IO[AnyStr], AnyStr]:
        file = super(LineStepper, self).unset_file()
        last_line = self.line
        self.line = None
        return file, last_line

    @property
    def is_running(self) -> bool:
        return self.workable and self.line is not None

    @property
    def line_offset(self) -> int | None:
        return self.file.tell() - len(self.line) if self.line else None

    def readline(self) -> AnyStr:
        self.line: AnyStr = self.file.readline()
        return self.line

    def step_forward(self) -> bool:
        self.line: AnyStr = self.file.readline()
        return bool(self.line)

    def step_back(self) -> bool:
        if self.line is None:
            raise ParsingError("Cannot determine how far to go, no line information")

        self.file.seek(self.file.tell() - len(self.line))
        self.line = None
        return self.file.tell() != 0

    def step_to(self, anchor_in: Predicate, /, *, check_last_read: bool = False) -> bool:
        if not self.workable:
            raise InvalidFile(f'File is not workable: {self.file!r}')

        if check_last_read and self.line:
            if anchor_in(self.line):
                return True

        while self.step_forward():
            if anchor_in(self.line):
                return True

        return self.handle_eof()

    def step_to_first(self, anchor_in: Predicate, until_in: Predicate, /, *, check_last_read: bool = False) -> bool:
        if not self.workable:
            raise InvalidFile(f'File is not workable: {self.file!r}')

        if check_last_read and self.line:
            if until_in(self.line):
                return False

            if anchor_in(self.line):
                return True

        while self.step_forward():
            if until_in(self.line):
                return False

            if anchor_in(self.line):
                return True

        return self.handle_eof()

    def step_until(self, anchors: list[Predicate], returns: list[Any], /, *, check_last_read: bool = False) -> bool | Any:
        if not self.workable:
            raise ValueError(f'File is not workable: {self.file!r}')

        if check_last_read and self.line:
            for anchor, r in zip(anchors, returns):
                if anchor(self.line):
                    return r

        while self.step_forward():
            for anchor, r in zip(anchors, returns):
                if anchor(self.line):
                    return r

        return self.handle_eof()

    def handle_eof(self) -> bool:
        if self.on_eof == 'raise':
            raise EOFReached(f'File {short_file_repr(self.file)!r} reached EOF')
        return False

    def __repr__(self) -> str:
        result = []
        if self.workable:
            if self.is_running:
                result.append(f'line={self.line[:10]!r}')
                result.append(f'line_offset={self.line_offset!r}')
            else:
                result.append(f'line={None!r}')

        result.append(f'file={short_file_repr(self.file)}')
        result.append(f'on_eof={self.on_eof!r}')
        result = ', '.join(result)
        return f'{self.__class__.__name__}({result})'

    @staticmethod
    def get_str_predicate(anchor: AnyStr | None) -> Predicate:
        if anchor is None:
            return lambda line: False  # Always False
        elif isinstance(anchor, (str, bytes)):
            return lambda line: anchor in line
        else:
            raise ValueError(f"Invalid anchor: {anchor!r}")


LineStepperType = TypeVar('LineStepperType', bound=LineStepper, covariant=True)


class TargetNotFound(ParsingError):
    def __init__(self, message: str, *args, line: str = '',
                 reached_eof: bool = False, n_steps: int | None = None) -> None:
        self.message = message
        self.line = line
        self.n_steps: int = n_steps
        self.reached_eof: bool = reached_eof
        super(TargetNotFound, self).__init__(self.get_message(), *args)

    def get_message(self) -> str:
        message = [self.message, f'(line={self.line!r}']
        if self.n_steps:
            message.append(f'(n_steps={self.n_steps!r})')
        message.append(f'(reached_eof={self.reached_eof})')
        return ' '.join(message)


class TerminatorReached(TargetNotFound):
    pass


class MaxStepsReached(TargetNotFound):
    pass


class RequirementNotSatisfied(ParsingError):
    pass


Predicate: TypeAlias = Callable[[AnyStr], bool]
Anchor: TypeAlias = AnyStr | None


class FileStepper(LineStepper):
    def target_not_found(
            self,
            message: str | None,
            n_steps: int | None = None,
            error_type: type[TargetNotFound] = TargetNotFound
    ) -> None:
        if message is not None:
            raise error_type(message,
                             line=self.line,
                             n_steps=n_steps,
                             reached_eof=self.end_of_file)

    def step_skip_n(self, n: int) -> bool:
        for _ in range(n):
            self.line = self.file.readline()
        return not self.end_of_file

    def step_apply_n(self, n: int, on_step: Processor) -> bool:
        for _ in range(n):
            self.line = self.file.readline()
            on_step(self.line)
        return not self.end_of_file

    def step_skip_until(self, until: Anchor | None,
                        check_last_read: bool = False) -> tuple[AnyStr, int]:
        target_in = self.get_str_predicate(until)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.line_offset

        while self.step_forward():
            if target_in(self.line):
                return self.line, self.line_offset

        self.target_not_found(message=f"Target '{until!r}' not found")

    def findall(self, target: Anchor, /, *, max_steps: int = -1, check_last_read: bool = False) -> list[AnyStr]:
        target_in = self.get_str_predicate(target)

        result: list[AnyStr] = []

        if check_last_read and self.line and target_in(self.line):
            result.append(self.line)

        n_steps = 0
        while self.step_forward():
            if target_in(self.line):
                result.append(self.line)

            if 0 <= max_steps <= n_steps:
                break

            n_steps += 1
        return result

    def finditer(self, target: Anchor, /, *, max_steps: int = -1, check_last_read: bool = False) -> Iterator[AnyStr, int]:
        target_in = self.get_str_predicate(target)

        if check_last_read and self.line and target_in(self.line):
            yield self.line, self.line_offset

        n_steps = 0
        while self.step_forward():
            if target_in(self.line):
                yield self.line, self.line_offset

            if 0 <= max_steps <= n_steps:
                break

            n_steps += 1

    def find(
            self,
            target: Anchor, /,
            max_steps: int = -1, *,
            check_last_read: bool = False,
            err_msg: str | None = None
    ) -> tuple[AnyStr, int] | None:
        target_in = self.get_str_predicate(target)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.file.tell()

        n_steps = 0
        while self.step_forward():
            if target_in(self.line):
                return self.line, self.line_offset

            if 0 <= max_steps <= n_steps:
                return self.target_not_found(message=err_msg, n_steps=n_steps,
                                             error_type=MaxStepsReached)
            n_steps += 1
        else:
            self.target_not_found(message=err_msg)

    def find_before(
            self,
            target: Anchor, terminator: Anchor, /,
            max_steps: int = -1, *,
            check_last_read: bool = False,
            err_msg: str | None = None
    ) -> tuple[AnyStr, int] | None:
        target_in = self.get_str_predicate(target)
        terminator_in = self.get_str_predicate(terminator)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.line_offset

        n_steps = 0
        while self.step_forward():
            if terminator_in(self.line):
                return self.target_not_found(message=err_msg, n_steps=n_steps,
                                             error_type=TerminatorReached)

            if target_in(self.line):
                return self.line, self.line_offset

            if 0 <= max_steps <= n_steps:
                return self.target_not_found(message=err_msg, n_steps=n_steps,
                                             error_type=MaxStepsReached)
            n_steps += 1
        else:
            self.target_not_found(message=err_msg)
