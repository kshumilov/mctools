from __future__ import annotations

import abc
import pathlib
import sys
import warnings

from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import partial
from typing import IO, Any, Callable, Optional, AnyStr, Iterable, Literal, TypeVar

from core.resources import Resources

from .error import ParsingError
from .resources import ResourceHandler
from .stepper import BaseFileStepper, BaseFileStepperType, Predicate, Anchor


__all__ = [
    'BaseFileParser',
    'FileParser',
    'CompositeParser',
    'SequentialParser',

    'ListenerSettings',
    'Listener',
    'DispatchFileParser',

    'Task',
    'TaskFileParser',

    # Types
    'Listeners',
    'Tasks',
]


class BaseFileParser(metaclass=abc.ABCMeta):
    OnError = Literal['skip', 'raise', 'warn']
    OnParsingErrorVariants: tuple[OnError, ...] = ('skip', 'raise', 'warn')

    on_parsing_error: OnError = 'raise'

    def __init__(self, on_parsing_error: OnError = 'raise') -> None:
        print('Initializing:', self.__class__.__name__, 'Base')
        super(BaseFileParser, self).__init__()

        if on_parsing_error not in BaseFileParser.OnParsingErrorVariants:
            raise ValueError(f'{self!r}: Unknown error handler: {on_parsing_error!r}')

        self.on_parsing_error = on_parsing_error

    def __repr__(self) -> str:
        params_repr = self._get_params_repr()
        return f'{self.__class__.__name__}({params_repr})'

    def _get_params_repr(self) -> str:
        return f'on_parsing_error={self.on_parsing_error!r}'

    def parse(self, file: pathlib.Path | IO[AnyStr], /, mode='rt', **kwargs) -> Any:
        print(f'{self!r}: Given {file!r}')
        try:
            if isinstance(file, pathlib.Path):
                filepath = pathlib.Path(file)
                with filepath.open(mode, **kwargs) as file:
                    data = self.parse_file(file)
            else:
                data = self.parse_file(file)
        except ParsingError as err:
            return self.handle_parsing_error(err)

        return data

    def parse_file(self, file: IO[AnyStr]) -> Any:
        print(f'{self!r}: Parsing')
        data = self._parse_file(file)
        print(f'{self!r}: Done parsing')
        return data

    @abc.abstractmethod
    def _parse_file(self, file: IO[AnyStr], /) -> Any:
        raise NotImplementedError

    def handle_parsing_error(self, err: ParsingError, /) -> Any:
        match self.on_parsing_error:
            case 'raise':
                raise err
            case 'warn':
                warnings.warn(err.args[0])


BaseFileParserType = TypeVar('BaseFileParserType', bound=BaseFileParser, covariant=True)


class FileParser(BaseFileParser, metaclass=abc.ABCMeta):
    DefaultStepper: type[BaseFileStepperType] = BaseFileStepper

    stepper: BaseFileStepperType | None = None

    def __init__(self, /, stepper: Optional[BaseFileStepperType] = None, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'File', **kwargs)
        super(FileParser, self).__init__(**kwargs)

        if isinstance(stepper, self.DefaultStepper):
            self.stepper = stepper
        else:
            self.stepper = self.DefaultStepper()

    @abc.abstractmethod
    def _parse_file(self, file: IO[AnyStr], /) -> Any:
        self.stepper.set_file(file)

    def _get_params_repr(self) -> str:
        base_repr = super(FileParser, self)._get_params_repr()
        return f'stepper={self.stepper!r}, {base_repr}'


FileParserType = TypeVar('FileParserType', bound=FileParser, covariant=True)


class CompositeParser(BaseFileParser, metaclass=abc.ABCMeta):
    def __init__(self, /, parser_parameters: dict | None = None, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'CompositeParser', **kwargs)
        super(CompositeParser, self).__init__(**kwargs)
        self.parser_parameters = parser_parameters or {}

    def build_parser(self, parser_cls: type[BaseFileParserType], /, *args, **kwargs) -> BaseFileParserType:
        print(f'{self!r}: Building {parser_cls.__name__!r}')
        parser_args = self.get_parser_args(parser_cls)
        parser_args.extend(args)

        parser_kwargs = self.get_parser_kwargs(parser_cls)
        parser_kwargs.update(kwargs)
        parser = parser_cls(*parser_args, **parser_kwargs)
        print(f'{self!r}: Done building  {parser!r}')
        return parser

    def get_parser_args(self, parser_cls: type[BaseFileParserType], /) -> list[Any]:
        return []

    def get_parser_kwargs(self, parser_cls: type[BaseFileParserType], /) -> dict[str, Any]:
        kwargs = self.parser_parameters.get(parser_cls, {})
        return kwargs

    def _get_params_repr(self) -> str:
        base_repr = super(CompositeParser, self)._get_params_repr()
        return f'parser_params={self.parser_parameters!r}, {base_repr}'


CompositeParserType = TypeVar('CompositeParserType', bound=CompositeParser, covariant=True)


class SequentialParser(CompositeParser, metaclass=abc.ABCMeta):
    Parsers: list[type[BaseFileParserType]] = []

    storage = None
    parsers = None

    def __init__(self, /, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'Sequential', **kwargs)
        super(SequentialParser, self).__init__(**kwargs)
        self.storage = []
        self.parsers: list[BaseFileParserType] = []

    def _parse_file(self, file: IO[AnyStr], /) -> Any:
        for parser in self.build_parsers():
            self.parsers.append(parser)
            data = parser.parse(file)
            self.storage.append(data)
        return self.clear_storage()

    def build_parsers(self) -> Iterable[BaseFileParserType]:
        for parser_cls in self.Parsers:
            yield self.build_parser(parser_cls)

    def clear_storage(self) -> list[Any]:
        result = []
        result.extend(self.storage)
        self.storage.clear()
        return result

    def _get_params_repr(self) -> str:
        params_repr = super(SequentialParser, self)._get_params_repr()
        parsers_repr = ','.join(f'{p.__name__}' for p in self.Parsers)
        storage_repr = ','.join(f'{k!r}' for k in self.storage)
        return f'parsers=[{parsers_repr!s}], storage={storage_repr!s}, {params_repr!s}'


Handler = Callable[[IO[AnyStr]], Any] | Callable[[], Any]


@dataclass(frozen=True, slots=True, eq=True, repr=True)
class ListenerSettings:
    max_runs: int = 1
    dispatch_file: bool = True
    dispatch_line: bool = False
    record_dispatches: bool = True


@dataclass
class Listener:
    label: str | Resources

    anchor: Anchor
    handle: Handler
    predicate: Predicate

    settings: ListenerSettings = field(default_factory=ListenerSettings)

    dispatch_offsets: list[int] = field(default_factory=list, init=False)
    was_dispatched: bool = field(default=False, init=False)

    def __post_init__(self):
        if 0 <= self.settings.max_runs:
            self.record_dispatches = True

    def run(self, stepper: BaseFileStepperType) -> Any:
        print(f'{self!r}: Handling {stepper!r}')

        self.was_dispatched = True
        if self.record_dispatches:
            self.record_dispatch(stepper)

        args = []
        if self.settings.dispatch_file:
            args.append(stepper.file)

        if self.settings.dispatch_line:
            args.append(stepper.line)

        data = self.handle(*args)
        print(f'{self!r}: Done handling {stepper!r}')
        return data

    def record_dispatch(self, stepper: BaseFileStepperType, /) -> None:
        self.dispatch_offsets.append(stepper.offset)

    @property
    def is_active(self) -> bool:
        return not (0 <= self.settings.max_runs <= self.n_runs)

    @property
    def n_runs(self) -> int | None:
        return len(self.dispatch_offsets) if self.record_dispatches else None

    def __repr__(self) -> str:
        max_runs = self.settings.max_runs
        if self.record_dispatches:
            dispatched = f'{self.n_runs}' + f'/{max_runs}' if max_runs > -1 else ''
        else:
            dispatched = '>= 1' if self.was_dispatched else '0'
        return f"{type(self).__name__}({self.label!s}, #Dispatched: {dispatched})"


Listeners = deque[Listener]


class DispatchFileParser(FileParser, metaclass=abc.ABCMeta):
    active_listeners = None
    inactive_listeners = None
    storage = None

    def __init__(self, /, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'DispatchFileParser', **kwargs)
        super(DispatchFileParser, self).__init__(**kwargs)
        self.storage = {}
        self.inactive_listeners: list[Listener] = []
        self.active_listeners = self.build_listeners()

    def _get_params_repr(self) -> str:
        params_repr = super(DispatchFileParser, self)._get_params_repr()
        active_repr = ','.join(f'{p!r}' for p in self.active_listeners) if self.active_listeners else None
        inactive_repr = ','.join(f'{p!r}' for p in self.inactive_listeners) if self.inactive_listeners else None
        return f'active=[{active_repr!s}], inactive=[{inactive_repr!s}], {params_repr!s}'

    def dispatch(self, line: AnyStr, /) -> bool:
        n_to_listen = len(self.active_listeners)
        while n_to_listen > 0 and (listener := self.active_listeners.pop()):
            if listener.predicate(line):
                self.storage[listener.label] = listener.run(self.stepper)

            if listener.is_active:
                self.active_listeners.appendleft(listener)
            else:
                self.inactive_listeners.append(listener)

            n_to_listen -= 1
        return not self.has_active_listeners

    def execute_dispatch(self) -> bool:
        print(f'{self!r}: Dispatching')
        data = self.stepper.step_to(self.dispatch)
        print(f'{self!r}: Done dispatching')
        return data

    def clear_storage(self) -> dict:
        result = {}
        result.update(self.storage)
        self.storage.clear()
        return result

    @property
    def total_dispatches(self):
        return sum(listener.n_runs for listener in self.active_listeners)

    @property
    def has_active_listeners(self) -> bool:
        return len(self.active_listeners) > 0

    @abc.abstractmethod
    def build_listeners(self) -> deque[Listener]:
        pass

    def _parse_file(self, file: IO[AnyStr]) -> Any:
        super(DispatchFileParser, self)._parse_file(file)
        self.execute_dispatch()
        return self.clear_storage()


@dataclass(frozen=True, repr=True, slots=True)
class Task:
    anchor: Anchor
    handle: str
    settings: ListenerSettings = field(default_factory=ListenerSettings)
    func_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_listener(self, label: str | Resources, stepper: BaseFileStepperType, /) -> Listener:
        handle = getattr(stepper, self.handle)
        if self.func_kwargs:
            handle = partial(handle, **self.func_kwargs)

        return Listener(
            label=label,
            anchor=self.anchor,
            predicate=stepper.get_predicate(self.anchor),
            handle=handle,
            settings=self.settings
        )


Tasks = dict[Resources, Task]


class TaskFileParser(DispatchFileParser, ResourceHandler, metaclass=abc.ABCMeta):
    DEFAULT_TASKS: dict[Resources, Task] = {}

    def __init__(self, /, tasks: Tasks | None = None, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'TaskFileParser', **kwargs)
        tasks = tasks or {}
        self.tasks: Tasks = self.DEFAULT_TASKS.copy()
        for resources, task in tasks.items():
            for resource in resources:
                if resource in self.DEFAULT_TASKS:
                    self.tasks[resource] = task
        super(TaskFileParser, self).__init__(**kwargs)

    def build_listeners(self) -> deque[Listener]:
        listeners = []
        for resource in self.resources:
            print(f'{self!r}: Building listener: {resource!r}')
            listener = self.assign_listener(resource)
            print(f'{self!r}: Done building: {listener!r}')
            listeners.append(listener)
        return deque(listeners)

    def get_default_task(self, resource: Resources, /) -> Task | None:
        for resources, task in self.tasks.items():
            if resource in resources:
                return task
        return None

    def assign_listener(self, resource: Resources, /, task: Task = None, **kwargs) -> Listener:
        task = task or self.get_default_task(resource)
        return task.get_listener(resource, self.stepper)
