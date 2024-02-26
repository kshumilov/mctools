import pathlib
from collections import defaultdict
from typing import Sequence

import click
from rich_click import RichCommand

from mctools.parsing.utils import parse_calculation, ParsingBackend
from mctools.parsing.gaussian.utils import GaussianSuffixes


__all__ = [
    'parse'
]


@click.command(
    name='parse',
    cls=RichCommand,
    help='Parsing utilities'
)
@click.argument(
    'filenames',
    type=click.Path(exists=True, path_type=pathlib.Path),
    nargs=-1,
)
@click.option('--include', multiple=True, type=click.Choice(list(GaussianSuffixes), case_sensitive=False), default=list(GaussianSuffixes))
@click.option('--ext', type=click.Path(path_type=pathlib.Path), required=False, default='.h5', help='Name of the hdf5 extenstion')
def parse(filenames: Sequence[pathlib.Path], include: tuple[str], ext: str) -> None:
    click.echo(filenames)
    selected = filter(lambda p: p.suffix in include, filenames)
    groups = defaultdict(list)
    for p in selected:
        groups[(p.parent, p.stem)].append(p)

    for (parent, stem), calc_filenames in groups.items():
        archive = parent / f'{stem}{ext}'
        parse_calculation(
            calc_filenames,
            backend=ParsingBackend.Gaussian,
            save=True,
            archivename=archive
        )
