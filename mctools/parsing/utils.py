import pathlib
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Sequence, TypeAlias

import h5py
import numpy as np

from mctools.cli.console import console
from mctools.core.resource import Resource
from mctools.newcore.storage import Storage

from mctools.core.basis import AtomicOrbitalBasis, MolecularOrbitalBasis

from .gaussian.utils import parse_gaussian_calc

__all__ = [
    'ParsingBackend',
    'parse_calculation',
    'save_data',
    'group_files',
]


class ParsingBackend(Enum):
    Gaussian = auto()


def save_data(storage: Storage, archive: pathlib.Path) -> None:
    with h5py.File(archive, 'w', libver='latest') as f:
        storage.to_hdf5(f)

        # for label, resource in result.items():
        #     name = '/'.join(label.name.split('_'))
        #
        #     if isinstance(resource, np.ndarray):
        #         if resource.ndim > 0:
        #             f.create_dataset(
        #                 name,
        #                 data=resource,
        #                 dtype=resource.dtype,
        #                 compression='gzip'
        #             )
        #         elif resource.ndim == 0:
        #             *path, name = label.name.split('_')
        #             path = '/'.join(path)
        #             gr = f.get(path) or f.create_group(path)
        #             gr.attrs[name] = resource


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
) -> dict[Resource, Any] | None:
    requested = requested or Resource.ALL()

    match backend:
        case ParsingBackend.Gaussian:
            console.rule('[bold red]Parsing Gaussian Calculation', style='bright_red')
            parsing_func = parse_gaussian_calc
        case _:
            raise ValueError(f'Unrecognized parsing backend')

    resources = parsing_func(filenames, requested)
    storage = Storage(resources=resources, consolidators=[
        AtomicOrbitalBasis, MolecularOrbitalBasis
    ])

    if save:
        archive = filenames[0].with_suffix(archive_ext)
        console.print(f'Saving to: {archive!r}')
        save_data(storage, archive=archive)
    console.rule('[bold red]Done', style='bright_red')

    if dont_return:
        return None
    return resources

