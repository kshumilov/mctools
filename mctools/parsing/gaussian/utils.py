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


def parse_gaussian_calc(filenames: Sequence[pathlib.Path], requested: Resource) -> dict[Resource, Any]:
    storage: dict[Resource, Any] = {}
    for filename in filenames:
        suffix = filename.suffix
        match GaussianSuffixes(suffix):
            case GaussianSuffixes.log:
                parser = LogParser(requested=requested)
            case GaussianSuffixes.fchk:
                parser = FchkParser(requested=requested)
            case _:
                console.print(f'Unrecognized file extension: {suffix}', style='red')
                continue

        with open(filename, 'r') as f:
            data, *_ = parser.parse(f)
            storage.update(data)

    return storage
