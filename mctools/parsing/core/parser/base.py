from __future__ import annotations

import abc
import io
import pathlib
import warnings
from typing import (
    TypeVar, TypeAlias, IO,
    AnyStr, Generic, Callable,
    ParamSpec, Concatenate, Any, cast, Literal
)

import attrs
from typing_extensions import overload

from ..error import ParsingError, ParserNotPrepared
from ..stepper import OnError
from ..filehandler import FileWithPosition

__all__ = [
    'Parser',
    'WrappingParser',
    'FWP',
]


FWP: TypeAlias = FileWithPosition[AnyStr]
I: TypeAlias = pathlib.Path | IO[AnyStr] | FWP[AnyStr]
Result = TypeVar('Result', covariant=True)  # Data from parse() type
Success: TypeAlias = tuple[Result, FWP[AnyStr]]
Failure: TypeAlias = tuple[None, FWP[AnyStr]]
ReturnType: TypeAlias = Success | Failure


@attrs.define(repr=True, eq=True)
class Parser(Generic[Result, AnyStr], metaclass=abc.ABCMeta):
    D = TypeVar('D')

    on_parsing_error: OnError = attrs.field(
        default='raise',
        validator=attrs.validators.in_(['skip', 'raise', 'warn'])
    )

    def parse(self, filelike: IO[AnyStr] | FWP[AnyStr], /) -> ReturnType[Result, AnyStr]:
        match filelike:
            case io.IOBase() as file:
                fwp = FileWithPosition(file)
                return self._parse_file(fwp)
            case FileWithPosition() as fwp:
                return self._parse_file(fwp)
            case _:
                raise ValueError('Invalid file')

    def _parse_file(self, file: FWP[AnyStr]) -> ReturnType[Result, AnyStr]:
        try:
            self.prepare(file)
            data, file = self.parse_file(file)
            return self.postprocess(data), self.cleanup(file)
        except ParsingError as err:
            return self.handle_error(err, file)

    def prepare(self, file: FWP[AnyStr], /) -> None:
        if not self.is_ready():
            raise ParserNotPrepared(f'Parser is not ready: {self!r}')

    def is_ready(self) -> bool:
        return True

    @abc.abstractmethod
    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[D, FWP[AnyStr]]:
        raise NotImplementedError

    def postprocess(self, raw_data: D, /) -> Result:
        return cast(Result, raw_data)

    def cleanup(self, file: FWP[AnyStr], /) -> FWP[AnyStr]:
        return file

    def handle_error(self, err: ParsingError, file: FWP[AnyStr]) -> tuple[None, FWP[AnyStr]]:
        match self.on_parsing_error:
            case 'raise':
                raise err
            case 'warn':
                warnings.warn(err.args[0])
        return None, file


Args = ParamSpec('Args')
Func = Callable[Concatenate[FWP[AnyStr], Args], tuple[Any, FWP[AnyStr]]]


@attrs.define(repr=True, eq=True)
class WrappingParser(Parser[Result, AnyStr], Generic[Result, AnyStr], metaclass=abc.ABCMeta):
    func: Func[AnyStr, Args] | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(attrs.validators.is_callable())
    )
    args: Args.args = attrs.field(factory=tuple)
    kwargs: Args.kwargs = attrs.field(factory=dict)

    def is_ready(self, /) -> bool:
        return (super(WrappingParser, self).is_ready() and
                self.func is not None)

    def parse_file(self, fwp: FWP[AnyStr], /) -> tuple[WrappingParser.D, FWP[AnyStr]]:
        return self.func(fwp, *self.args, **self.kwargs)

    @classmethod
    def from_parser(cls, other: Parser[Result, AnyStr], /, *args: Args.args, **kwargs: Args.kwargs) -> WrappingParser[Result, AnyStr]:
        kwargs.pop('func', None)
        return cls(func=other.parse, args=args, kwargs=kwargs)
