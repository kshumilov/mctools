import pathlib
import multiprocessing
from typing import Sequence

import click
from rich_click import RichCommand, RichGroup

from mctools.parsing.utils import (
    parse_calculation,
    group_files,
    CalculationGroups,
    ParsingBackend
)


__all__ = [
    'parse'
]


def sequential_parse(groups: CalculationGroups, ext: str) -> None:
    for group_filenames in groups.values():
        parse_calculation(group_filenames, archive_ext=ext)


@click.command(
    name='gaussian',
    cls=RichCommand,
    help='Parse a Gaussian Output files such Log, Fchk, and RWFDump'
)
@click.option(
    '-j', '--cpu', 'n_cpu',
    type=int,
    default=multiprocessing.cpu_count(),
    help='Number of processes to use for parsing multiple calculations',
    required=False,
    show_default=True,
)
@click.argument(
    'filenames',
    type=click.Path(exists=True, path_type=pathlib.Path),
    nargs=-1,
)
@click.option(
    '--ext',
    type=str,
    required=False,
    default='.h5',
    show_default=True,
    help='Name of the archive extension',
)
def gaussian(filenames: Sequence[pathlib.Path], n_cpu: int, ext: str,) -> None:
    groups = group_files(filenames)

    if len(groups) <= 1 or n_cpu == 1:
        sequential_parse(groups, ext)
        return

    with multiprocessing.Pool(n_cpu) as pool:
        pool.starmap(parse_calculation, [
            (g, None, ParsingBackend.Gaussian, True, ext, True)
            for g in groups.values()
        ])


@click.group(
    cls=RichGroup,
    name='parse',
    help='Parsing utilities'
)
def parse():
    pass


parse.add_command(gaussian)