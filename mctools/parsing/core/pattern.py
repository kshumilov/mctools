from __future__ import annotations

import re

from collections import namedtuple
from dataclasses import dataclass, field, InitVar, replace
from typing import Generic, ClassVar, TypeVar, Any, Callable, Iterator, AnyStr


__all__ = [
    'ProcessedPattern',
]


grouped_tmplt = r'(?P<%s>%s)'

simple_float_tmplt = r'[+\-]?\d*\.\d*'
simple_int_tmplt = r'[+\-]?\d+'
simple_bool_tmplt = r'[tTfF]'

int_patt = grouped_tmplt % (r'%s', simple_int_tmplt)
float_patt = r'(?P<%s>%s([DeE]%s)?)' % (r'%s', simple_float_tmplt, simple_int_tmplt)
bool_patt = grouped_tmplt % (r'%s', simple_bool_tmplt)

ParsingResultType = dict[str, Any]


@dataclass(slots=True)
class ProcessedPattern(Generic[AnyStr]):
    ReturnType: ClassVar[TypeVar] = TypeVar('ReturnType')
    MatchDict = dict[str | int, Any]
    GroupMap = Callable[[str], Any]
    GroupMaps = dict[str, GroupMap]

    MatchFunc = Callable[[MatchDict], Any]
    MatchFuncs = dict[str, MatchFunc]

    pattern: re.Pattern[AnyStr] | AnyStr

    # A function, named tuple or a name for a named tuple to store the information extracted from the matched pattern
    constructor: Callable[..., tuple] | str | dict | None = field(default=None, repr=False)

    group_maps: GroupMaps = field(default_factory=dict)
    default_group_map: GroupMap = str

    match_funcs: MatchDict | MatchFunc = field(default_factory=dict)

    flags: InitVar[re.RegexFlag] = re.UNICODE
    unnamed_group_prefix: str = field(default='', repr=False)

    named_groups: dict[int, str] = field(init=False, repr=False)

    def __post_init__(self, flags: re.RegexFlag | None = re.UNICODE) -> None:
        if not isinstance(self.pattern, re.Pattern):
            self.pattern: re.Pattern[AnyStr] = re.compile(self.pattern, flags=flags)

        if self.pattern.groups < 1:
            raise ValueError(f'ProcessedPattern must have at least one group to be useful,'
                             f'use re.Pattern instead: {self.pattern}')

        self.named_groups = {v: k for k, v in self.pattern.groupindex.items()}

        if self.constructor is None:
            self.constructor: type[dict[str, Any]] = dict
        elif isinstance(self.constructor, str):
            if self.include_unnamed_groups:
                group_keys = [self._get_group_key(i)
                              for i in range(1, self.pattern.groups + 1)]
            else:
                group_keys = list(self.named_groups.values())

            self.constructor = namedtuple(
                self.constructor, group_keys
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
        result: ProcessedPattern[AnyStr].MatchDict = {}

        if self.include_unnamed_groups:
            group_keys = range(1, self.pattern.groups + 1)
        else:
            group_keys = self.named_groups.keys()

        for idx in group_keys:
            key = self._get_group_key(idx)
            func = self.group_maps.get(key, self.default_group_map)
            result[key] = func(match.group(idx))

        return result

    def transform_match(self, match_dict: MatchDict):
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
