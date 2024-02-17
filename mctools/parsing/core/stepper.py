from __future__ import annotations


from typing import IO, AnyStr, Callable, Any, Iterator, TypeVar

from .error import ParsingError

__all__ = [
    # Classes
    'BaseFileStepper',
    'FileStepper',

    # Types
    'BaseFileStepperType',
    'Predicate',
    'Anchor',
]


class FileHandler:
    file: IO[AnyStr] | None = None
    eof_line: str | bytes = ''

    def set_eof_line(self, file: IO[AnyStr]) -> None:
        self.eof_line = b'' if 'b' in file.mode else ''

    def set_file(self, file: IO[AnyStr] | None) -> None:
        self.file = file
        self.set_eof_line(file)

    def __repr__(self) -> str:
        file_repr = f'file={self.file.name!r}'
        return f'{self.__class__.__name__}({file_repr})'


Processor = Callable[[AnyStr], bool | Any]


class BaseFileStepper(FileHandler):
    line: AnyStr | None = None

    def __repr__(self) -> str:
        file_repr = f'file={self.file!r}'
        if self.file and self.line:
            file_repr += f', line={self.line[:15]!r} offset={self.offset!r}'
        return f'{self.__class__.__name__}({file_repr})'

    def set_file(self, file: IO[AnyStr] | None, line: AnyStr = None) -> None:
        super(BaseFileStepper, self).set_file(file)
        self.line = line

    @property
    def has_parsable_file(self) -> bool:
        return not self.file.closed and self.file.readable() and self.file.seekable()

    @property
    def end_of_file(self) -> bool:
        return self.line == self.eof_line

    @property
    def is_running(self) -> bool:
        return self.line and self.line != self.eof_line

    @property
    def offset(self) -> int | None:
        return self.file.tell() - len(self.line) if self.line else None

    def readline(self) -> AnyStr:
        self.line: AnyStr = self.file.readline()
        return self.line

    def step_forward(self) -> bool:
        self.line: AnyStr = self.file.readline()
        return not self.end_of_file

    def step_back(self) -> bool:
        self.file.seek(self.file.tell() - len(self.line))
        self.line = None
        return not self.end_of_file

    def check_file(self):
        if not self.has_parsable_file:
            raise ParsingError(f"{self!r}: File {self.file!r} is not parsable")

    def step_to(self, anchor_in: Predicate, /, *, check_last_read: bool = False) -> bool:
        self.check_file()

        if check_last_read and self.line:
            if anchor_in(self.line):
                return True

        while self.step_forward():
            if anchor_in(self.line):
                return True

        return False

    def step_to_first(self, anchor_in: Predicate, until_in: Predicate, /, *, check_last_read: bool = False) -> bool:
        self.check_file()

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

        return False

    def step_until(self, anchors: list[Predicate], returns: list[Any], /, *, check_last_read: bool = False) -> bool | Any:
        self.check_file()

        if check_last_read and self.line:
            for anchor, r in zip(anchors, returns):
                if anchor(self.line):
                    return r

        while self.step_forward():
            for anchor, r in zip(anchors, returns):
                if anchor(self.line):
                    return r

        return False

    @staticmethod
    def get_predicate(anchor: Anchor | None) -> Predicate:
        if anchor is None:
            return lambda l: False  # Always False
        elif isinstance(anchor, (str, bytes)):
            return lambda line: anchor in line
        elif callable(anchor):
            return anchor
        else:
            raise ValueError(f"Invalid anchor: {anchor!r}")

    @staticmethod
    def always_false(line: str) -> bool:
        return False


BaseFileStepperType = TypeVar('BaseFileStepperType', bound=BaseFileStepper, covariant=True)


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


Predicate = Callable[[AnyStr], bool]
Anchor = AnyStr | Predicate


class FileStepper(BaseFileStepper):
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
        target_in = self.get_predicate(until)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.offset

        while self.step_forward():
            if target_in(self.line):
                return self.line, self.offset

        self.target_not_found(message=f"Target '{until!r}' not found")

    def findall(self, target: Anchor, /, *, max_steps: int = -1, check_last_read: bool = False) -> list[AnyStr]:
        target_in = self.get_predicate(target)

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
        target_in = self.get_predicate(target)

        if check_last_read and self.line and target_in(self.line):
            yield self.line, self.offset

        n_steps = 0
        while self.step_forward():
            if target_in(self.line):
                yield self.line, self.offset

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
        target_in = self.get_predicate(target)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.file.tell()

        n_steps = 0
        while self.step_forward():
            if target_in(self.line):
                return self.line, self.offset

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
        target_in = self.get_predicate(target)
        terminator_in = self.get_predicate(terminator)

        if check_last_read and self.line and target_in(self.line):
            return self.line, self.offset

        n_steps = 0
        while self.step_forward():
            if terminator_in(self.line):
                return self.target_not_found(message=err_msg, n_steps=n_steps,
                                             error_type=TerminatorReached)

            if target_in(self.line):
                return self.line, self.offset

            if 0 <= max_steps <= n_steps:
                return self.target_not_found(message=err_msg, n_steps=n_steps,
                                             error_type=MaxStepsReached)
            n_steps += 1
        else:
            self.target_not_found(message=err_msg)
