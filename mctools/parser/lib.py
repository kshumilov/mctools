import re

from typing import TextIO, Callable, Any, NoReturn, Optional

__all__ = [
    'transform_match_groups',
    'transform_match',
    'MatchDict', 'GroupMap', 'MatchFunc',

    'find_line_in_file',
    'find_pattern_in_line',
    'find_pattern_in_file',

    'grouped_tmplt',
    'simple_int_tmplt',
    'simple_float_tmplt',
    'simple_bool_tmplt',
]


MatchDict = dict[str, Any]
GroupMap = Callable[[str], Any]
MatchFunc = Callable[[MatchDict], Any]


grouped_tmplt = r'(?P<%s>%s)'
simple_float_tmplt = r'[+\-]?\d*\.\d*'
simple_int_tmplt = r'[+\-]?\d+'
simple_bool_tmplt = r'[tTfF]'


def transform_match_groups(match: re.Match, group_maps: Optional[dict[str, GroupMap]] = None,
                           default_group_map: GroupMap = str) -> MatchDict:
    group_maps = group_maps if group_maps else {}
    
    result: MatchDict = {}
    match_dict = match.groupdict()
    for group_name, value in match_dict.items():
        func = group_maps.get(group_name, default_group_map)
        result[group_name] = func(value)
    
    return result


def transform_match(match_dict: MatchDict, match_funcs: Optional[dict[str, MatchFunc]] = None) -> NoReturn:
    match_funcs = match_funcs if match_funcs else {}
    for key, func in match_funcs.items():
        match_dict[key] = func(match_dict)
        
        
def find_line_in_file(file: TextIO, patt: re.Pattern, first_line: Optional[str] = '',
                      group_maps: Optional[dict[str, GroupMap]] = None, default_group_map: GroupMap = str,
                      match_funcs: Optional[dict[str, MatchFunc]] = None) -> tuple[Optional[MatchDict], str]:
    while line := (first_line if first_line else file.readline()):
        if match := patt.search(line):
            match_dict = transform_match_groups(match, group_maps=group_maps, default_group_map=default_group_map)
            transform_match(match_dict, match_funcs=match_funcs)
            return match_dict, line
        first_line = ''
    else:
        return None, line
    

def find_pattern_in_line(line: str, patt: re.Pattern,
                         group_maps: Optional[dict[str, GroupMap]] = None, default_group_map: GroupMap = str,
                         match_funcs: Optional[dict[str, MatchFunc]] = None) -> list[MatchDict]:
    results: list[MatchDict] = []
    for match in patt.finditer(line):
        match_dict = transform_match_groups(match, group_maps=group_maps, default_group_map=default_group_map)
        transform_match(match_dict, match_funcs)
        results.append(match_dict)
    return results


def find_pattern_in_file(file: TextIO, patt: str | re.Pattern, first_line: Optional[str] = '',
                         max_matches: int = 1, until_first_failure: bool = False, n_skips: int = 0,
                         group_maps: Optional[dict[str, GroupMap]] = None, default_group_map: GroupMap = str,
                         match_funcs: Optional[dict[str, MatchFunc]] = None) -> tuple[MatchDict | list[MatchDict], str]:
    # Initialize pattern
    patt: re.Pattern = patt if isinstance(patt, re.Pattern) else re.compile(patt)
    
    results: list[MatchDict] = []
    
    # Run through the rest of the file line by line
    n_failed_matches = 0
    while ((not (until_first_failure and (n_failed_matches - n_skips) > 0)) and
           (len(results) < max_matches) and
           (line := (first_line if first_line else file.readline()))):

        line_results = find_pattern_in_line(line, patt, group_maps=group_maps,
                                            default_group_map=default_group_map, match_funcs=match_funcs)
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
