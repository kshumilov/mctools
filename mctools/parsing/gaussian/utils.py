import pathlib

from enum import StrEnum
from typing import Sequence, Any

from mctools.cli.console import console
from mctools.core.resource import Resource

from . import LogParser, FchkParser

__all__ = [
    'GaussianSuffixes',
    'parse_gaussian_calc',
]


class GaussianSuffixes(StrEnum):
    log = '.log'
    fchk = '.fchk'
    chk = '.chk'
    ci = '.ci'


def parse_gaussian_calc(filenames: Sequence[pathlib.Path], requested: Resource) -> dict[Resource, Any]:
    storage: dict[Resource, Any] = {}
    for filename in filenames:
        suffix = filename.suffix
        match suffix:
            case GaussianSuffixes.log.value:
                parser = LogParser(requested=requested)
            case GaussianSuffixes.fchk.value:
                parser = FchkParser(requested=requested)
            case GaussianSuffixes.ci.value | GaussianSuffixes.chk.value:
                console.print(f'Parsing of {suffix} is not implemented', style='red')
                continue
            case _:
                console.print(f'Unrecognized suffix {suffix}', style='red')
                continue

        with open(filename, 'r') as f:
            data, *_ = parser.parse(f)
            storage.update(data)

    return storage
