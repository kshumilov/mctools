from typing import Any

import h5py
import numpy as np

from mctools.core.mcspace import MCSpace
from mctools.core.resource import Resource

__all__ = [
    'save_data',
]


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
