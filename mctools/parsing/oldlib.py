from __future__ import annotations

import inspect
import re
import warnings
from typing import IO, TextIO, Callable, Any

from .core import ProcessedPattern
from .core.parser import TargetNotFound
from .core.pattern import ParsingResultType


def search_in_file(
        file: IO, patt: str | bytes | re.Pattern | ProcessedPattern, /,
        n_skips: int = -1, *, first_line: str = '',
        err_msg: str | None = None
) -> tuple[ProcessedPattern.ReturnType | re.Match | None, str]:
    patt = re.compile(patt) if not isinstance(patt, (re.Pattern, ProcessedPattern)) else patt

    line = first_line if first_line else file.readline()
    while line:
        if (match := patt.search(line)) is not None:
            return match, line

        if n_skips == 0:
            if err_msg is not None:
                raise TargetNotFound(err_msg, line=line)
            return None, line

        n_skips -= 1
        line = file.readline()

    if err_msg is not None:
        raise TargetNotFound(err_msg, line=line, reached_eof=True)

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
                    raise TargetNotFound(err_msg, line=line)
                return results, line
            n_skips -= 1
        else:
            n_skips = max_skips

        line = file.readline()

    return results, line


def parse_calc_name(filename: str, *patterns: re.Pattern | ProcessedPattern) -> tuple[list[bool], ParsingResultType]:
    result: ParsingResultType = {}
    matched: list[bool] = [False] * len(patterns)
    for i, pattern in enumerate(patterns):
        if matches := pattern.findall(filename):
            result.update(matches.pop())
            matched[i] = True
    return matched, result


def parse_file(file: TextIO, read_funcs: list[Callable], /, result: ParsingResultType, *, first_line: str = ''):
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
                        args.append(value(result) if callable(value) else value)
                    else:
                        print(f'Skipping: {func.__name__}: positional parameter {param} is unavailable')
                        break
                elif value is not None and info.kind == info.POSITIONAL_OR_KEYWORD:
                    kwargs[param] = value(result) if callable(value) else value
        else:
            try:
                print(f'Executing: {func.__name__}', end='')
                data, line = func(file, *args, **kwargs, first_line=line)
                result.update(data)
                print(' --- Done')
            except TargetNotFound as err:
                print(' --- Could not find pattern')
                warnings.warn(*err.args)

    return result, line
