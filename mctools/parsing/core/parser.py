from __future__ import annotations

import abc
import pathlib
import sys
import warnings

import attr

from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from functools import partial
from typing import IO, Any, Callable, Optional, AnyStr, Iterable, Literal, TypeVar, Generic, ClassVar, TypeAlias, Type, \
    NamedTuple, MutableMapping

from core.resources import Resources

from .error import ParsingError, ParserNotPrepared
from .resources import ResourceHandler
from .stepper import LineStepper, Predicate, Anchor


__all__ = [
    'Parser',
    'FileParser',
    'CompositeParser',
    'SequentialParser',

    'ListenerConfig',
    'Listener',
    'DispatchFileParser',

    'Task',
    'TaskFileParser',

    # Types
    'Listeners',
    'Tasks',
]


R = TypeVar('R')  # Data Return type
OnError: TypeAlias = Literal['skip', 'raise', 'warn']


@attr.define(repr=True, eq=True, kw_only=True)
class Parser(Generic[R], metaclass=abc.ABCMeta):
    @attr.define(repr=True, eq=True, kw_only=True)
    class Config:
        on_parsing_error: OnError = attr.field(
            default='raise', kw_only=True,
            validator=attr.validators.instance_of(str),
        )

        def is_ready(self) -> bool:
            return True

    config: Config = attr.field(
        default=attr.NOTHING,
        factory=attr.Factory(
            lambda self: type(self).Config(),
            takes_self=True,
        ),
    )

    @classmethod
    def from_config(cls, config: Config) -> 'Parser[R]':
        return cls(config=config)

    def configure(self, new_config: Config) -> bool:
        self.config = attr.evolve(
            self.config, **attr.asdict(
                new_config,
                filter=attr.filters.exclude(type(None)),
            )
        )
        return self.config.is_ready()

    def update_config(self, **kwargs) -> None:
        self.config = attr.evolve(self.config, **kwargs)

    def parse(self, file: pathlib.Path | IO[AnyStr], /, mode='rt', **kwargs) -> tuple[R, tuple[IO[R], AnyStr] | tuple]:
        print(f'Given {file!r}')
        try:
            if isinstance(file, pathlib.Path):
                filepath = pathlib.Path(file)
                with filepath.open(mode, **kwargs) as file:
                    data, *_ = self._parse_file(file)
                    return data, tuple()
            else:
                return self._parse_file(file)
        except ParsingError as err:
            return self.handle_parsing_error(err)

    def _parse_file(self, file: IO[AnyStr]) -> R:
        if not self.prepare(file):
            raise ParserNotPrepared(f'Not ready for parsing: {self!r}')

        print(f'Parsing: {self!r}')
        data = self.parse_file(file)
        return self.cleanup(data, file)

    def prepare(self, file: IO[AnyStr], /) -> bool:
        print(f'Preparing: {self!r}')
        return self.config.is_ready()

    @abc.abstractmethod
    def parse_file(self, file: IO[AnyStr], /) -> Any:
        raise NotImplementedError

    def cleanup(self, data: R, file: IO[AnyStr]) -> R:
        print(f'{self!r}: Done parsing')
        return data

    def handle_parsing_error(self, err: ParsingError, /) -> Any:
        match self.config.on_parsing_error:
            case 'raise':
                raise err
            case 'warn':
                warnings.warn(err.args[0])


@attr.define(repr=True, eq=True)
class FileParser(Parser[R], metaclass=abc.ABCMeta):
    @attr.define(repr=True, eq=True, kw_only=True)
    class Config(Parser.Config):
        stepper: LineStepper = attr.field(
            factory=LineStepper,
            validator=attr.validators.instance_of(LineStepper),
        )

        def is_ready(self) -> bool:
            ready = super(FileParser.Config, self).is_ready()
            return ready and self.stepper.workable

    @classmethod
    def from_stepper(cls, stepper: LineStepper) -> 'FileParser':
        return cls(config=cls.Config(stepper=stepper))

    def prepare(self, file: IO[AnyStr]) -> bool:
        self.stepper.set_file(file)
        is_ready = super(FileParser, self).prepare(file)
        return is_ready & self.config.is_ready()

    def cleanup(self, data: R, file: IO[AnyStr]) -> tuple[R, tuple[IO[AnyStr], str]]:
        file, last_line = self.stepper.unset_file()
        return data, (file, last_line)

    @property
    def stepper(self) -> LineStepper:
        return self.config.stepper


@attr.define(repr=True, eq=True, slots=True, hash=True)
class ParserClassKey:
    name: str = attr.field(validator=attr.validators.instance_of(str))
    index: int = attr.field(default=0, validator=attr.validators.ge(0))

    @classmethod
    def from_parser_class(cls, parser_class: type[Parser], /, index: int = 0) -> ParserClassKey:
        return cls(parser_class.__name__, index=index)

    @classmethod
    def convert_key(cls, key: type[Parser] | tuple[type[Parser], int] | ParserClassKey) -> ParserClassKey:
        match key:
            case (parser_class, int(index)) if isinstance(parser_class, Parser):
                return cls.from_parser_class(parser_class, index)
            case cls():
                raise key
            case parser_class if issubclass(parser_class, Parser):
                return cls.from_parser_class(parser_class)
            case _:
                raise KeyError(f"Invalid config_key: {key!r}")


@attr.define(repr=False, eq=True)
class ParserBuilder(MutableMapping[ParserClassKey, Parser.Config]):
    configs: dict[ParserClassKey, Parser.Config] = attr.field(
        factory=lambda: defaultdict(dict),
        validator=attr.validators.deep_mapping(
            key_validator=attr.validators.instance_of(ParserClassKey),
            value_validator=attr.validators.instance_of(Parser.Config),
            mapping_validator=attr.validators.instance_of(dict),
        )
    )

    def __getitem__(self, key: ParserClassKey | tuple[type[Parser], int] | type[Parser]) -> Parser.Config:
        key = ParserClassKey.convert_key(key)
        return self.configs[key]

    def __setitem__(self, key: ParserClassKey, value) -> None:
        key = ParserClassKey.convert_key(key)
        self.configs[key] = value

    def __delitem__(self, key: ParserClassKey) -> None:
        key = ParserClassKey.convert_key(key)
        del self.configs[key]

    def __len__(self):
        return len(self.configs)

    def __iter__(self):
        return iter(self.configs)


ParserType = TypeVar('ParserType', bound=Parser, covariant=True)


@attr.define(repr=True, eq=True)
class CompositeParser(Parser[R], metaclass=abc.ABCMeta):
    @attr.define(repr=True, eq=True, kw_only=True)
    class Config(Parser.Config):
        parsers_configs: ParserBuilder = attr.Factory(ParserBuilder)

    parsers: dict[ParserClassKey, Parser] = attr.Factory(dict)
    counter: defaultdict[str, int] = attr.Factory(lambda: defaultdict(int))

    def prepare(self, file: IO[AnyStr], /) -> bool:
        is_ready = super(CompositeParser, self).prepare(file)
        self.build_parsers()
        return is_ready and self.config.is_ready()

    def build_parsers(self) -> None:
        for parser_cls in self.get_parser_classes():
            config_key = self.generate_config_key(parser_cls)
            self.build_parser(parser_cls, config_key)

    def get_parser_classes(self) -> Iterable[type[Parser[R]]]:
        return []

    def generate_config_key(self, parser_class: type[Parser[R]], /) -> ParserClassKey:
        name = parser_class.__name__
        index = self.counter[name]
        self.counter[name] += 1
        config_key = ParserClassKey.from_parser_class(parser_class, index)
        return config_key

    def build_parser(self, parser_class: type[ParserType[R]], /, config_key: ParserClassKey | int = 0) -> Parser[R]:
        if isinstance(config_key, int):
            config_key = ParserClassKey.from_parser_class(parser_class, config_key)

        config = self.config.parsers_config[parser_class]
        parser = parser_class.from_parser_class(config)
        self.parsers[config_key] = parser
        return parser

    def run_parser(self, file: IO[AnyStr], parser: Parser[R], config_key: ParserClassKey | int = 0) -> R:
        if isinstance(config_key, int):
            config_key = ParserClassKey.from_parser_class(parser.__class__, config_key)

        if not parser.config.is_ready():
            self.configure_parser(parser, config_key)

        data = parser.parse(file)
        self.update_config(data, parser, config_key)
        return data

    def configure_parser(self, parser: Parser[R], config_key: ParserClassKey | int = 0) -> None:
        if isinstance(config_key, int):
            config_key = ParserClassKey.from_parser_class(parser.__class__, config_key)

        parser_config = self.config.parsers_config[config_key]
        parser.configure(parser_config)

    def update_config(self, data: R, parser: Parser[R], config_key: ParserClassKey, /) -> None:
        pass


@attr.define(repr=True)
class SequentialParser(CompositeParser[R], metaclass=abc.ABCMeta):
    Parsers: ClassVar[list[type[Parser]]] = []

    @attr.define(repr=True, eq=True, kw_only=True)
    class Config(Parser.Config):
        parsers_configs: ParserBuilder = attr.Factory(ParserBuilder)

    storage: list[R] = attr.Factory(list)

    def __attrs_post_init__(self):
        if any(not issubclass(parser_class, Parser) for parser_class in self.Parsers):
            raise ValueError(f'Classes in {type(self).__name__} must inherit from Parser')

    def get_parser_classes(self) -> Iterable[type[Parser]]:
        for parser_cls in self.Parsers:
            yield parser_cls

    def parse_file(self, file: IO[AnyStr], /) -> Any:
        for config_key, parser in self.parsers.items():
            data = self.run_parser(file, parser, config_key)
            self.storage.append(data)
        return self.storage

    def cleanup(self, data: R, file: IO[AnyStr]) -> list[R]:
        result = data.copy()
        self.storage.clear()
        return result


Handler = Callable[[IO[AnyStr]], Any] | Callable[[], Any]


@attr.define(slots=True, eq=True, repr=True)
class ListenerConfig:
    max_runs: int = attr.field(default=1, validator=attr.validators.instance_of(int))
    dispatch_file: bool = attr.field(default=True, validator=attr.validators.instance_of(bool))
    dispatch_line: bool = attr.field(default=False, validator=attr.validators.instance_of(bool))
    record_dispatches: bool = attr.field(default=True, validator=attr.validators.instance_of(bool))

    def __attrs_post_init__(self):
        if 0 <= self.max_runs:
            self.record_dispatches = True


@attr.define(slots=True, eq=True, repr=False)
class Listener:
    label: str = attr.field(validator=attr.validators.instance_of(str))

    handle: Handler = attr.field(validator=attr.validators.is_callable())
    predicate: Predicate = attr.field(validator=attr.validators.is_callable())

    config: ListenerConfig

    anchor: Anchor | None = attr.field(
        default=None,
        validator=attr.validators.optional([
            attr.validators.instance_of(str),
            attr.validators.instance_of(bytes),
        ])
    )

    dispatch_offsets: list[int] = attr.field(factory=list, init=False)
    was_dispatched: bool = attr.field(
        default=False,
        init=False,
        validator=attr.validators.instance_of(bool),
    )

    def run(self, stepper: LineStepper) -> Any:
        print(f'{self!r}: Handling {stepper!r}')

        self.was_dispatched = True
        if self.config.record_dispatches:
            self.record_dispatch(stepper)

        args = []
        if self.config.dispatch_file:
            args.append(stepper.file)

        if self.config.dispatch_line:
            args.append(stepper.line)

        data = self.handle(*args)
        print(f'{self!r}: Done handling {stepper!r}')
        return data

    def record_dispatch(self, stepper: LineStepper, /) -> None:
        self.dispatch_offsets.append(stepper.line_offset)

    @property
    def is_active(self) -> bool:
        return not (0 <= self.config.max_runs <= self.n_runs)

    @property
    def n_runs(self) -> int | None:
        return len(self.dispatch_offsets) if self.config.record_dispatches else None

    def __repr__(self) -> str:
        max_runs = self.config.max_runs
        if self.config.record_dispatches:
            dispatched = f'{self.n_runs}' + f'/{max_runs}' if max_runs > -1 else ''
        else:
            dispatched = '>= 1' if self.was_dispatched else '0'
        return f"{type(self).__name__}({self.label!s}, #Dispatched: {dispatched})"


Listeners = deque[Listener]


@attr.define(repr=True, eq=True)
class DispatchFileParser(FileParser[R], metaclass=abc.ABCMeta):
    active_listeners: deque[Listener] | None = attr.field(
        default=None,
        validator=attr.validators.optional(
            attr.validators.deep_iterable(
                member_validator=attr.validators.instance_of(Listener),
                iterable_validator=attr.validators.instance_of(deque),
            )))
    inactive_listeners: list[Listener] = attr.Factory(list)
    storage: defaultdict[str, list[R]] = attr.Factory(lambda: defaultdict(list))

    def prepare(self, file: IO[AnyStr]) -> bool:
        super(DispatchFileParser, self).prepare(file)
        self.build_listeners()

    def dispatch(self, line: AnyStr, /) -> bool:
        n_to_listen = len(self.active_listeners)
        while n_to_listen > 0 and (listener := self.active_listeners.pop()):
            if listener.predicate(line):
                self.storage[listener.label].append(listener.run(self.stepper))

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

    def cleanup(self, data: R, file: IO[AnyStr]) -> list[R]:
        result = data.copy()
        self.storage.clear()
        return result

    @property
    def total_dispatches(self):
        return sum(listener.n_runs for listener in self.active_listeners)

    @property
    def has_active_listeners(self) -> bool:
        return len(self.active_listeners) > 0

    def build_listeners(self):
        pass

    def parse_file(self, file: IO[AnyStr]) -> R:
        return self.execute_dispatch()


@attr.define(attr=True, slots=True, repr=True)
class Task:
    anchor: Anchor = attr.field(validator=)
    handle: str = attr.field(validator=attr.validators.instance_of(str))
    settings: ListenerConfig = field(default_factory=ListenerConfig)
    handle_kwargs: dict[str, Any] = field(default_factory=dict)

    def get_listener(self, label: str | Resources, stepper: LineStepper, /) -> Listener:
        handle = getattr(stepper, self.handle)
        if self.handle_kwargs:
            handle = partial(handle, **self.handle_kwargs)

        return Listener(
            label=label,
            anchor=self.anchor,
            predicate=stepper.get_str_predicate(self.anchor),
            handle=handle,
            config=self.settings,
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
