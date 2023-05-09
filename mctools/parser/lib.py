import math
import re
import inspect
import warnings

from typing import TextIO, Callable, Any, NoReturn, Optional

__all__ = [
    'ParsingResult',

    'transform_match_groups',
    'transform_match',
    'MatchDict', 'GroupMap', 'MatchFunc',
    'GroupMaps', 'MatchFuncs',

    'find_line_in_file',
    'find_pattern_in_line',
    'find_pattern_in_file',

    'parse_calc_name',
    'parse_file',

    'grouped_tmplt',
    'simple_int_tmplt',
    'simple_float_tmplt',
    'simple_bool_tmplt',
]


ParsingResult = dict[str, Any]

MatchDict = dict[str, Any]
GroupMap = Callable[[str], Any]
GroupMaps = dict[str, GroupMap]

MatchFunc = Callable[[MatchDict], Any]
MatchFuncs = dict[str, MatchFunc]


grouped_tmplt = r'(?P<%s>%s)'
simple_float_tmplt = r'[+\-]?\d*\.\d*'
simple_int_tmplt = r'[+\-]?\d+'
simple_bool_tmplt = r'[tTfF]'


def transform_match_groups(match: re.Match, group_maps: Optional[GroupMaps] = None,
                           default_group_map: GroupMap = str, unnamed_group_prefix: str = 'g') -> MatchDict:
    group_maps = group_maps if group_maps else {}
    
    result: MatchDict = {}
    match_dict = match.groupdict()
    if not match_dict:
        match_dict.update({
            f'{unnamed_group_prefix}{i}': v
            for i, v in enumerate(match.groups())
        })

    for group_name, value in match_dict.items():
        func = group_maps.get(group_name, default_group_map)
        result[group_name] = func(value)
    
    return result


def transform_match(match_dict: MatchDict, match_funcs: Optional[MatchFuncs] = None) -> NoReturn:
    match_funcs = match_funcs if match_funcs else {}
    for key, func in match_funcs.items():
        match_dict[key] = func(match_dict)
        
        
def find_line_in_file(file: TextIO, patt: re.Pattern, first_line: Optional[str] = '', n_skips: int = -1,
                      group_maps: Optional[GroupMaps] = None, default_group_map: GroupMap = str,
                      match_funcs: Optional[MatchFuncs] = None,
                      unnamed_group_prefix: str = 'g') -> tuple[Optional[MatchDict], str]:
    while line := (first_line if first_line else file.readline()):
        if match := patt.search(line):
            match_dict = transform_match_groups(match, group_maps=group_maps, default_group_map=default_group_map,
                                                unnamed_group_prefix=unnamed_group_prefix)
            transform_match(match_dict, match_funcs=match_funcs)
            return match_dict, line

        if n_skips == 0:
            return None, line

        n_skips -= 1
        first_line = ''
    else:
        return None, line
    

def find_pattern_in_line(line: str, patt: re.Pattern,
                         group_maps: Optional[dict[str, GroupMap]] = None, default_group_map: GroupMap = str,
                         match_funcs: Optional[dict[str, MatchFunc]] = None,
                         unnamed_group_prefix: str = 'g') -> list[MatchDict]:
    results: list[MatchDict] = []
    for match in patt.finditer(line):
        match_dict = transform_match_groups(match, group_maps=group_maps, default_group_map=default_group_map,
                                            unnamed_group_prefix=unnamed_group_prefix)
        transform_match(match_dict, match_funcs)
        results.append(match_dict)
    return results


def find_pattern_in_file(file: TextIO, patt: str | re.Pattern, first_line: Optional[str] = '',
                         max_matches: int | float = 1, until_first_failure: bool = False, n_skips: int = 0,
                         group_maps: Optional[GroupMaps] = None, default_group_map: GroupMap = str,
                         match_funcs: Optional[MatchFuncs] = None,
                         unnamed_group_prefix: str = 'g') -> tuple[MatchDict | list[MatchDict], str]:
    # Initialize pattern
    patt: re.Pattern = patt if isinstance(patt, re.Pattern) else re.compile(patt)
    
    results: list[MatchDict] = []
    
    # Run through the rest of the file line by line
    n_failed_matches = 0
    while ((not (until_first_failure and (n_failed_matches - n_skips) > 0)) and
           (len(results) < max_matches) and
           (line := (first_line if first_line else file.readline()))):

        line_results = find_pattern_in_line(line, patt,
                                            group_maps=group_maps,
                                            default_group_map=default_group_map,
                                            match_funcs=match_funcs,
                                            unnamed_group_prefix=unnamed_group_prefix)
        if not line_results:
            n_failed_matches += 1
        else:
            n_failed_matches = 0
        
        if (n := len(results) + len(line_results)) > max_matches:
            excess = n - max_matches
            line_results = line_results[:len(line_results) - excess]
            
        results.extend(line_results)
        first_line = ''
        
    if max_matches == 1 and results:
        return results[0], line
        
    return results, line


def parse_calc_name(filename: str, patterns: list[re.Pattern], /,
                    group_maps: Optional[GroupMaps] = None, default_group_map: GroupMap = str,
                    match_funcs: Optional[MatchDict] = None,
                    unnamed_group_prefix: str = 'g') -> dict[str, int | float | str]:
    info = {}
    for patt in patterns:
        if (matches := find_pattern_in_line(filename, patt,
                                            group_maps=group_maps, default_group_map=default_group_map,
                                            match_funcs=match_funcs, unnamed_group_prefix=unnamed_group_prefix)):
            info.update(matches.pop())
    return info


def parse_file(file: TextIO, read_funcs: list[Callable], result: ParsingResult, /, first_line: str = ''):
    line = first_line

    for func in read_funcs:
        args = []
        for param, info in inspect.signature(func).parameters.items():
            if param in {'file', 'first_line'}:
                continue
            elif info.kind == info.POSITIONAL_OR_KEYWORD or info.kind == info.KEYWORD_ONLY:
                continue

            # TODO: generalize to read kwargs too
            if param in result:
                args.append(result[param])
            else:
                print(f'Skipping: {func.__name__}, parameter {param} is unavailable')
                break
        else:
            try:
                print(f'Executing: {func.__name__}', end='')
                data, line = func(file, *args, first_line=line)
                result.update(data)
                print(' --- Done')
            except ValueError as err:
                print(' --- Issued Warning')
                warnings.warn(*err.args)

    return result, line
