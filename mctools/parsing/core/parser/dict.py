from __future__ import annotations

from collections import deque, defaultdict
from typing import TypeVar, Hashable, AnyStr, TypeAlias, Generic, Any, ClassVar

import attrs

from ..error import AnchorNotFound
from .base import Parser
from ..stepper import LineStepper, Anchor, Predicate
from ..filehandler import FileWithPosition

__all__ = [
    'Listener',
    'DispatchParser',
]

F: TypeAlias = FileWithPosition[AnyStr]
R = TypeVar('R')  # Data from parse() type

Label = TypeVar('Label', bound=Hashable, contravariant=True)


@attrs.define(slots=True, eq=True, repr=False)
class Listener(Parser[R, AnyStr], Generic[Label, R, AnyStr]):
    parser: Parser[R, AnyStr] | None = attrs.field(
        factory=Parser,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(Parser),
        )
    )

    anchor: AnyStr | None = attrs.field(
        default=None,
        validator=attrs.validators.optional([
            attrs.validators.instance_of((str, bytes)),
        ]),
    )

    predicate: Predicate[AnyStr] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.is_callable(),
        )
    )

    label: Label | None = attrs.field(
        validator=attrs.validators.optional(
            attrs.validators.instance_of(Hashable)
        ),
    )

    max_runs: int = attrs.field(
        default=1,
        validator=attrs.validators.instance_of(int),
    )

    dispatch_offsets: list[int] = attrs.field(factory=list, init=False)

    @label.default
    def _get_default_label(self) -> Label | None:
        if self.anchor is not None:
            return str(self.anchor)

        if self.parser is not None:
            return type(self.parser).__name__

        if self.predicate is not None:
            return self.predicate.__name__

        return None

    # @predicate.default
    # def _get_default_predicate(self) -> Predicate[AnyStr] | None:
    #     if self.anchor is not None:
    #         return LineStepper().get_anchor_predicate(self.anchor)
    #     return None

    def is_ready(self) -> bool:
        return (super(Listener, self).is_ready() and
                self.parser is not None and
                self.predicate is not None and
                self.label is not None)

    def parse_file(self, fwp: F, /) -> tuple[Any, F]:
        self.record_dispatch(fwp)
        return self.parser.parse(fwp)

    def record_dispatch(self, file: F, /) -> None:
        self.dispatch_offsets.append(file.n_lines_read)

    @property
    def is_active(self) -> bool:
        return not (0 <= self.max_runs <= self.n_runs)

    @property
    def n_runs(self) -> int:
        return len(self.dispatch_offsets)

    def __repr__(self) -> str:
        dispatched = f'{self.n_runs}' + f'/{self.max_runs}' if self.max_runs > -1 else ''
        return f"{type(self).__name__}({self.label!s}, #Dispatched: {dispatched})"


@attrs.define(repr=True, eq=True)
class DispatchParser(Parser[dict[Label, list[R]], AnyStr]):
    stepper: LineStepper[AnyStr] = attrs.field(
        factory=LineStepper,
        validator=attrs.validators.instance_of(LineStepper),
    )
    terminator: Anchor | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(str)
        )
    )

    active_listeners: deque[Listener] = attrs.field(
        factory=deque,
        init=False,
    )
    inactive_listeners: list[Listener] = attrs.field(
        factory=deque,
        init=False,
    )
    storage: defaultdict[Label, list[Any]] = attrs.field(
        factory=lambda: defaultdict(list),
        init=False,
    )

    def cleanup(self, file: F) -> F:
        self.storage.clear()
        self.active_listeners.clear()
        return super(DispatchParser, self).cleanup(file)

    def parse_file(self, fwp: F) -> tuple[dict[Label, list[Any]], F]:
        self.stepper.take(fwp)

        if self.terminator is not None:
            terminator_in = self.stepper.get_anchor_predicate(self.terminator)
            found_all = self.stepper.step_until(self.on_step, terminator_in)
        else:
            found_all = self.stepper.step_to(self.on_step)

        if not found_all:
            raise AnchorNotFound(f'Active listeners [{self.active_listeners}] '
                                 f'did not find any targets')

        return self.storage.copy(), self.stepper.return_file()

    def on_step(self, line: AnyStr, /) -> bool:
        n_to_listen = len(self.active_listeners)
        while n_to_listen > 0:
            listener = self.active_listeners.pop()

            if listener.predicate is None:
                listener.predicate = self.stepper.get_anchor_predicate(listener.anchor)

            if listener.predicate(line):
                file = self.stepper.return_file()
                data, file = listener.parse(file)
                self.stepper.take(file)

                self.storage[listener.label].append(data)

            if listener.is_active:
                self.active_listeners.appendleft(listener)
            else:
                self.inactive_listeners.append(listener)

            n_to_listen -= 1
        return not self.has_listeners

    def add_listener(self, listener: Listener):
        self.active_listeners.appendleft(listener)

    @property
    def total_dispatches(self):
        return sum(listener.n_runs for listener in self.active_listeners)

    @property
    def has_listeners(self) -> bool:
        return len(self.active_listeners) > 0
