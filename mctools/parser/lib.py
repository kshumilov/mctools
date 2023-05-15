from __future__ import annotations

import re
import inspect
import warnings

from collections import namedtuple
from dataclasses import dataclass, field, InitVar, replace

from typing import TextIO, Callable, Any, NoReturn, TypeVar, ClassVar, Iterator, AnyStr

__all__ = [
    'ParsingResult',
    'MatchDict', 'GroupMap', 'MatchFunc',
    'GroupMaps', 'MatchFuncs',

    'ProcessedPattern',

    'search_in_file',
    'findall_in_file',

    'parse_calc_name',
    'parse_file',

    'grouped_tmplt',
    'simple_int_tmplt',
    'simple_float_tmplt',
    'simple_bool_tmplt',
    'int_patt',
    'float_patt',
    'bool_patt',

    'bool_map',

    'PatternNotFound',
]

ParsingResult = dict[str, Any]

MatchDict = dict[str | int, Any]
GroupMap = Callable[[str], Any]
GroupMaps = dict[str, GroupMap]

MatchFunc = Callable[[MatchDict], Any]
MatchFuncs = dict[str, MatchFunc]

grouped_tmplt = r'(?P<%s>%s)'
simple_float_tmplt = r'[+\-]?\d*\.\d*'
simple_int_tmplt = r'[+\-]?\d+'
simple_bool_tmplt = r'[tTfF]'

int_patt = grouped_tmplt % (r'%s', simple_int_tmplt)
float_patt = r'(?P<%s>%s([DeE]%s)?)' % (r'%s', simple_float_tmplt, simple_int_tmplt)
bool_patt = grouped_tmplt % (r'%s', simple_bool_tmplt)


def bool_map(x: str) -> bool:
    return x.capitalize() == 'T'


class PatternNotFound(Exception):
    def __init__(self, message: str, /, line: str = '', to_end: bool = False):
        self.line = line
        self.to_end = to_end

        if self.to_end:
            message += f' (to_end={self.to_end})'

        super(PatternNotFound, self).__init__(message)

    @property
    def message(self) -> str:
        return self.args[0]


@dataclass(slots=True)
class ProcessedPattern:
    ReturnType: ClassVar[TypeVar] = TypeVar('ReturnType')

    pattern: re.Pattern | AnyStr
    constructor: Callable[[Any], ReturnType] | str | None = field(default=None, repr=False)

    group_maps: GroupMaps = field(default_factory=dict)
    default_group_map: GroupMap = str

    match_funcs: MatchDict | MatchFunc = field(default_factory=dict)

    flags: InitVar[re.RegexFlag] = re.UNICODE
    unnamed_group_prefix: str = field(default='', repr=False)

    named_groups: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self, flags: re.RegexFlag | None = re.UNICODE):
        if not isinstance(self.pattern, re.Pattern):
            self.pattern = re.compile(self.pattern, flags=flags)

        if self.pattern.groups < 1:
            raise ValueError(f'ProcessedPattern must have at least one group to be useful,'
                             f'use re.Pattern instead: {self.pattern}')

        self.named_groups = {v: k for k, v in self.pattern.groupindex.items()}

        if self.constructor is None:
            self.constructor = dict
        elif isinstance(self.constructor, str):
            if self.include_unnamed_groups:
                group_keys = [self._get_group_key(i)
                              for i in range(1, self.pattern.groups + 1)]
            else:
                group_keys = list(self.named_groups.values())

            self.constructor = namedtuple(
                self.constructor,
                group_keys + list(self.match_funcs.keys())
            )

    def match(self, *args, **kwargs) -> ReturnType | None:
        if match := self.pattern.match(*args, **kwargs):
            return self.process(match)
        return None

    def fullmatch(self, *args, **kwargs) -> ReturnType | None:
        if match := self.pattern.fullmatch(*args, **kwargs):
            return self.process(match)
        return None

    def search(self, *args, **kwargs) -> ReturnType | None:
        if match := self.pattern.search(*args, **kwargs):
            return self.process(match)
        return None

    def finditer(self, *args, **kwargs) -> Iterator[ReturnType]:
        for match in self.pattern.finditer(*args, **kwargs):
            yield self.process(match)

    def findall(self, *args, **kwargs) -> list[ReturnType]:
        return list(self.finditer(*args, **kwargs))

    def transform_groups(self, match: re.Match) -> MatchDict:
        result: MatchDict = {}

        if self.include_unnamed_groups:
            group_keys = range(1, self.pattern.groups + 1)
        else:
            group_keys = self.named_groups.keys()

        for idx in group_keys:
            key = self._get_group_key(idx)
            func = self.group_maps.get(key, self.default_group_map)
            result[key] = func(match.group(idx))

        return result

    def transform_match(self, match_dict: MatchDict) -> NoReturn:
        for key, func in self.match_funcs.items():
            match_dict[key] = func(match_dict)

    def process(self, match: re.Match) -> tuple | ReturnType:
        match_dict = self.transform_groups(match)
        self.transform_match(match_dict)
        return self.constructor(**match_dict)

    def update_pattern(self, patt: AnyStr) -> 'ProcessedPattern':
        new_patt = re.compile(self.pattern.pattern % patt, flags=self.pattern.flags)
        return replace(self, pattern=new_patt)

    @property
    def include_unnamed_groups(self) -> bool:
        return bool(self.unnamed_group_prefix)

    def _get_group_key(self, i: int) -> str:
        return self.named_groups.get(i, f'{self.unnamed_group_prefix}{i}')


def search_in_file(file: TextIO, patt: str | re.Pattern | ProcessedPattern, /,
                   n_skips: int = -1, *, first_line: str = '',
                   err_msg: str | None = None) -> tuple[ProcessedPattern.ReturnType | re.Match | None, str]:
    patt = re.compile(patt) if not isinstance(patt, (re.Pattern, ProcessedPattern)) else patt

    line = first_line if first_line else file.readline()
    while line:
        if (match := patt.search(line)) is not None:
            return match, line

        if n_skips == 0:
            if err_msg is not None:
                raise PatternNotFound(err_msg, line=line)
            return None, line

        n_skips -= 1
        line = file.readline()

    if err_msg is not None:
        raise PatternNotFound(err_msg, line=line, to_end=True)

    return None, line


def findall_in_file(file: TextIO, patt: str | re.Pattern | ProcessedPattern, /,
                    max_matches: int | float = 1, max_skips: int = 0, keep_excess: bool = True, *,
                    first_line: str = '', err_msg: str | None = None
                    ) -> tuple[list[tuple | ProcessedPattern.ReturnType] | None, str]:
    patt = re.compile(patt) if not isinstance(patt, (re.Pattern, ProcessedPattern)) else patt

    n_skips = max_skips
    results: list[ProcessedPattern.ReturnType | tuple] = []
    line = first_line if first_line else file.readline()
    while line and len(results) < max_matches:
        line_results = patt.findall(line)
        if not keep_excess and (n := len(results) + len(line_results)) > max_matches:
            excess = n - max_matches
            line_results = line_results[:len(line_results) - excess]

        results.extend(line_results)

        if not line_results:
            if n_skips == 0:
                if err_msg is not None:
                    raise PatternNotFound(err_msg, line=line)
                return results, line
            n_skips -= 1
        else:
            n_skips = max_skips

        line = file.readline()

    return results, line


def parse_calc_name(filename: str, patterns: list[re.Pattern | ProcessedPattern], /) -> dict[str, int | float | str]:
    info = {}
    for patt in patterns:
        matches = patt.findall(filename)
        info.update(matches.pop())
    return info


def parse_file(file: TextIO, read_funcs: list[Callable], /, result: ParsingResult, *, first_line: str = ''):
    line = first_line

    for func in read_funcs:
        args: list[Any] = []
        kwargs: dict[str, Any] = {}
        for param, info in inspect.signature(func).parameters.items():
            if param in {'file', 'first_line'} or info.kind == info.KEYWORD_ONLY:
                continue
            else:
                value = result.get(param)
                if info.kind == info.POSITIONAL_ONLY:
                    if value is not None:
                        args.append(value)
                    else:
                        print(f'Skipping: {func.__name__}: positional parameter {param} is unavailable')
                        break
                elif value is not None and info.kind == info.POSITIONAL_OR_KEYWORD:
                    kwargs[param] = value
        else:
            try:
                print(f'Executing: {func.__name__}', end='')
                data, line = func(file, *args, **kwargs, first_line=line)
                result.update(data)
                print(' --- Done')
            except PatternNotFound as err:
                print(' --- Could not find pattern')
                warnings.warn(*err.args)

    return result, line
