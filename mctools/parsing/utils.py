import pathlib
from enum import Enum, auto
from typing import Any, Sequence

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
]


class ParsingBackend(Enum):
    Gaussian = auto()


def save_data(result: dict[Resource, Any], archive_name: str) -> None:
    with h5py.File(archive_name, 'w', libver='latest') as f:
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


def parse_calculation(
        filename: str | Sequence[str],
        requested: Resource | None = None,
        backend: ParsingBackend = ParsingBackend.Gaussian,
        save: bool = True,
        archivename: str = '',
) -> dict[Resource, Any]:
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


    if not save:
        return storage

    console.print(f'Saving to: {archivename!r}')
    archivename = archivename or filenames[0].with_suffix('.h5')
    save_data(storage, archive_name=archivename)

    console.rule('[bold red]Done', style='bright_red')

    return storage
