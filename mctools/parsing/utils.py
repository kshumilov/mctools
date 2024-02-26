import pathlib
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Sequence, TypeAlias

import h5py
import numpy as np

from mctools.cli.console import console
from mctools.core.mcspace import MCSpace
from mctools.core.resource import Resource

from .gaussian.utils import parse_gaussian_calc

__all__ = [
    'ParsingBackend',
    'parse_calculation',
    'save_data',
    'group_calculations',
]


class ParsingBackend(Enum):
    Gaussian = auto()


def save_data(result: dict[Resource, Any], archive: pathlib.Path) -> None:
    with h5py.File(archive, 'w', libver='latest') as f:
        for label, resource in result.items():
            name = '/'.join(label.name.split('_'))

            if isinstance(resource, np.ndarray):
                if resource.ndim > 0:
                    f.create_dataset(
                        name,
                        data=resource,
                        dtype=resource.dtype,
                        compression='gzip'
                    )

                if resource.ndim == 0:
                    *path, name = label.name.split('_')
                    path = '/'.join(path)
                    gr = f.get(path) or f.create_group(path)
                    gr.attrs[name] = resource

            if isinstance(resource, MCSpace):
                ds = f.create_dataset(name, data=resource.graph.spaces)
                ds.attrs['max_hole'] = resource.graph.max_hole
                ds.attrs['max_elec'] = resource.graph.max_elec


CalculationGroups: TypeAlias = defaultdict[tuple[pathlib.Path, str], list[pathlib.Path]]


def group_calculations(filenames: Sequence[pathlib.Path]) -> CalculationGroups:
    groups = defaultdict(list)
    for p in filenames:
        groups[(p.parent, p.stem)].append(p)
    return groups


def parse_calculation(
        filename: pathlib.Path | Sequence[pathlib.Path],
        requested: Resource | None = None,
        backend: ParsingBackend = ParsingBackend.Gaussian,
        save: bool = True,
        archive_ext: str = '.h5',
        dont_return: bool = False
) -> dict[Resource, Any] | None:
    requested = requested or Resource.ALL()

    match filename:
        case [*filenames]:
            filenames = [pathlib.Path(f) for f in filenames]
        case str(filename):
            filenames = [pathlib.Path(filename)]
        case _:
            raise ValueError(f'Invalid filename format: {filename}')

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
        save_data(storage, archive=archive)
    console.rule('[bold red]Done', style='bright_red')

    if dont_return:
        return None
    return storage

