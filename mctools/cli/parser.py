import pathlib

import click

from mctools.parsing.gaussian import LogParser, FchkParser
from mctools.parsing.utils import save_data


__all__ = [
    'parse'
]

LOG_SUFFIX = 'log'
FCHK_SUFFIX = 'fchk'
GAUSSIAN_SUFFIXES = [LOG_SUFFIX, FCHK_SUFFIX]


@click.command(name='parse')
@click.argument('calc', type=click.Path(exists=True, path_type=pathlib.Path))
@click.option('--include', multiple=True, type=click.Choice(GAUSSIAN_SUFFIXES, case_sensitive=False), default=GAUSSIAN_SUFFIXES)
@click.option('--archive', type=click.Path(path_type=pathlib.Path), required=False, help='Name of the hdf5 archive: by default [CALC].h5')
def parse(calc: pathlib.Path, include: tuple[str], archive: str | None = None) -> None:
    data = {}
    if LOG_SUFFIX in include:
        filename = calc.with_suffix(f'.{LOG_SUFFIX}')
        with open(filename, 'r') as f:
            click.echo(f'Parsing {LOG_SUFFIX.capitalize()}: {filename}')

            parser = LogParser()
            (route, result), *_ = parser.parse(f)
            data.update(result)

    if FCHK_SUFFIX in include:
        filename = calc.with_suffix(f'.{FCHK_SUFFIX}')
        with open(filename, 'r') as f:
            click.echo(f'Parsing {FCHK_SUFFIX.capitalize()}: {filename}')
            parser = FchkParser()
            result, *_ = parser.parse(f)
            data.update(result)

    if archive is None:
        archive = calc.with_suffix('.h5')

    save_data(data, archive)
