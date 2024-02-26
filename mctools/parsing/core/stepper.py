from __future__ import annotations

import warnings
from typing import AnyStr, Literal, TypeAlias, Callable

import attrs

from .filehandler import FileHandler, DoesNotHaveAFile, is_handler_type
from .error import EOFReached, TerminatorReached


__all__ = [
    'LineStepper',
    'OnError',

    'Predicate',
]


OnError: TypeAlias = Literal['raise', 'skip', 'warn']
Predicate: TypeAlias = Callable[[AnyStr], bool]
PredicateFactory: TypeAlias = Callable[[AnyStr], Predicate[AnyStr]]


def build_in_predicate(anchor: AnyStr) -> Predicate[AnyStr]:
    def predicate(line: AnyStr) -> bool:
        return anchor in line

    return predicate


def build_startswith_predicate(anchor: AnyStr) -> Predicate[AnyStr]:
    def predicate(line: AnyStr) -> bool:
        return line.startswith(anchor)

    return predicate


@attrs.define(repr=True)
class LineStepper(FileHandler[AnyStr]):
    @property
    def is_file_binary(self) -> bool:
        return is_handler_type(self, bytes)

    def readline(self, /) -> AnyStr:
        if self.fwp is not None:
            self.fwp.forward()
            return self.fwp.last_line
        raise DoesNotHaveAFile()

    def step(self, /) -> bool:
        if self.fwp is not None:
            self.fwp.forward()
            return bool(self.fwp.last_line)
        raise DoesNotHaveAFile()

    def step_back(self, /) -> bool:
        if self.fwp is not None:
            return self.fwp.backward()
        raise DoesNotHaveAFile()

    def step_to(self,
                anchor_in: Predicate[AnyStr], /, *,
                check_last_read: bool = False,
                on_eof: OnError = 'skip') -> bool:
        if self.fwp is None:
            raise DoesNotHaveAFile()

        if check_last_read and self.fwp.last_line:
            if anchor_in(self.fwp.last_line):
                return True

        while self.fwp.forward():
            if anchor_in(self.fwp.last_line):
                return True

        self._handle_on_eof(on_eof)
        return False

    def step_until(
            self,
            anchor_in: Predicate[AnyStr],
            terminator_in: Predicate[AnyStr], /, *,
            check_last: bool = False,
            on_eof: OnError = 'skip',
            on_terminator: OnError = 'skip',
    ) -> bool:
        if self.fwp is None:
            raise DoesNotHaveAFile()

        if check_last and self.fwp.last_line:
            if terminator_in(self.fwp.last_line):
                self._handle_on_terminator(on_terminator)
                return False

            if anchor_in(self.fwp.last_line):
                return True

        while self.fwp.forward():
            if terminator_in(self.fwp.last_line):
                self._handle_on_terminator(on_terminator)
                return False

            if anchor_in(self.fwp.last_line):
                return True

        self._handle_on_eof(on_eof)
        return False

    def _handle_on_eof(self, on_eof: OnError) -> None:
        match on_eof:
            case 'raise':
                raise EOFReached()
            case 'warn':
                warnings.warn('EOF reached')

    def _handle_on_terminator(self, on_terminator: OnError) -> None:
        match on_terminator:
            case 'raise':
                raise TerminatorReached()
            case 'warn':
                warnings.warn('Terminator reached')

    def get_anchor_predicate(
            self, anchor: str, /,
            func: PredicateFactory[AnyStr] = build_in_predicate
    ) -> Predicate[str] | Predicate[bytes]:
        if is_handler_type(self, bytes):
            return func(anchor.encode('utf-8'))
        elif is_handler_type(self, str):
            return func(anchor)
        else:
            raise ValueError(f'Unrecognized handler type: {self!r}')


    # def findall(self, target: Anchor, /, *, check_last_read: bool = False) -> list[AnyStr]:
    #     target_in = self.get_str_predicate(target)
    #
    #     result: list[AnyStr] = []
    #
    #     if check_last_read and self.line and target_in(self.line):
    #         result.append(self.line)
    #
    #     n_steps = 0
    #     while self.step_forward():
    #         if target_in(self.line):
    #             result.append(self.line)
    #
    #         if 0 <= max_steps <= n_steps:
    #             break
    #
    #         n_steps += 1
    #     return result
    #
    # def finditer(self, target: Anchor, /, *, max_steps: int = -1, check_last_read: bool = False) -> Iterator[AnyStr, int]:
    #     target_in = self.get_str_predicate(target)
    #
    #     if check_last_read and self.line and target_in(self.line):
    #         yield self.line, self.line_offset
    #
    #     n_steps = 0
    #     while self.step_forward():
    #         if target_in(self.line):
    #             yield self.line, self.line_offset
    #
    #         if 0 <= max_steps <= n_steps:
    #             break
    #
    #         n_steps += 1
