import pathlib
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Sequence, TypeAlias

from mctools.cli.console import console
from mctools.newcore.resource import Resource
from mctools.newcore.storage import Storage

from .gaussian.utils import parse_gaussian_calc

__all__ = [
    'ParsingBackend',
    'parse_calculation',
    'group_files',
]


class ParsingBackend(Enum):
    Gaussian = auto()


CalculationGroups: TypeAlias = defaultdict[tuple[pathlib.Path, str], list[pathlib.Path]]


def group_files(filenames: Sequence[pathlib.Path]) -> CalculationGroups:
    groups = defaultdict(list)
    for p in filenames:
        groups[(p.parent, p.stem)].append(p)
    return groups


def parse_calculation(
        filenames: Sequence[pathlib.Path],
        requested: Resource | None = None,
        backend: ParsingBackend = ParsingBackend.Gaussian,
        save: bool = True,
        archive_ext: str = '.h5',
        dont_return: bool = False
) -> Storage | None:
    requested = requested or Resource.ALL()

    match backend:
        case ParsingBackend.Gaussian:
            console.rule('[bold red]Parsing Gaussian Calculation', style='bright_red')
            parsing_func = parse_gaussian_calc
        case _:
            raise ValueError(f'Unrecognized parsing backend')

    storage = parsing_func(filenames, requested)

    if save:
        archive = filenames[0].with_suffix(archive_ext)
        console.print(f'Saving to: {archive!r}')
        storage.to_hdf5(archive)

    console.rule('[bold red]Done', style='bright_red')

    if dont_return:
        return None

    return storage

