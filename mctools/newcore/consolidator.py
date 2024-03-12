from __future__ import annotations

import abc
import pathlib
import types
from typing import Any, ClassVar, TypeAlias, cast, get_args, Union, get_origin, TypeVar

import attrs
import h5py
import numpy as np
import pandas as pd

from mctools.cli.console import console

from mctools.newcore.resource import Resource
from .metadata import MCTOOLS_METADATA_KEY

__all__ = [
    'Archived',
    'Resourced',
    'Consolidator',
]

Storage: TypeAlias = dict[Resource, Any]


def is_union(t: object) -> bool:
    origin = get_origin(t)
    return origin is Union or origin is types.UnionType


T = TypeVar('T', bound=type)


def get_union_subclasses(union: Union | types.UnionType, base: T | tuple[T, ...]) -> list[T]:
    return list(filter(lambda t: issubclass(t, base), get_args(union)))


@attrs.define(repr=True, eq=True)
class Archived(metaclass=abc.ABCMeta):
    TO_KEY: ClassVar[str] = 'to_hdf5'
    ROOT_KEY: ClassVar[str] = 'hdf5_root'
    NAME_KEY: ClassVar[str] = 'hdf5_name'
    IGNORE_KEY: ClassVar[str] = 'ignore_hdf5'

    ROOT: ClassVar[str] = ''

    def to_hdf5(self, filename: pathlib.Path, /, prefix: str = '') -> None:
        attrs.resolve_types(type(self))
        for attr_name, attribute in attrs.fields_dict(type(self)).items():
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, {})

            if metadata.get(self.IGNORE_KEY, False) or not attribute.init:
                continue

            elif to_hdf5 := getattr(self, f'{attr_name}_{self.TO_KEY}', None):
                to_hdf5(filename, prefix=prefix)

            elif (value := getattr(self, attr_name)) is not None and hasattr(value, self.TO_KEY):
                value.to_hdf5(filename, prefix)

            else:
                path, name = self.get_attr_hdf5_path(attr_name)
                if to_hdf5 := metadata.get(self.TO_KEY):
                    value = to_hdf5(value)

                if isinstance(value, np.ndarray):
                    with h5py.File(filename, 'a') as file:
                        file.require_dataset('/'.join([path, name]), data=value, shape=value.shape, dtype=value.dtype)
                    del file
                elif value is not None:
                    with h5py.File(filename, 'a') as file:
                        gr = file.require_group(path)
                        gr.attrs[name] = value
                    del file
                else:
                    continue

    @classmethod
    def from_hdf5(cls, filename: pathlib.Path, /, prefix: str = '') -> Archived:
        args = []
        kwargs = {}

        attrs.resolve_types(cls)
        for attr_name, attribute in attrs.fields_dict(cls).items():
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, {})
            if metadata.get(cls.IGNORE_KEY, False) or not attribute.init:
                continue

            elif from_hdf5 := getattr(cls, f'{attr_name}_from_hdf5', None):
                value = from_hdf5(filename, prefix=prefix)

            elif isinstance(attribute.type, type) and issubclass(attribute.type, Archived):
                value = attribute.type.from_hdf5(filename, prefix=prefix)

            elif is_union(attribute.type) and (bases := get_union_subclasses(attribute.type, Archived)):
                # TODO: apply each base to until first successful one
                value = bases[0].from_hdf5(filename, prefix=prefix)
            else:
                path, name = cls.get_attr_hdf5_path(attr_name, prefix=prefix)
                with h5py.File(filename, 'r') as file:
                    if isinstance(node := file.get('/'.join([path, name])), h5py.Dataset):
                        value = np.asarray(node)
                    elif isinstance(node := file.get(path), h5py.Group) and name in node.attrs:
                        value = node.attrs.get(name)
                    else:
                        continue

            if attribute.kw_only or attribute.default is not attrs.NOTHING:
                kwargs[attr_name] = value
            else:
                args.append(value)

        return cls(*args, **kwargs)

    @classmethod
    def get_root_hdf5_path(cls, prefix: str = '') -> str:
        return '/'.join([cls.ROOT, prefix])

    @classmethod
    def get_attr_hdf5_path(cls, attr_name: str, /, prefix: str = '') -> tuple[str, str]:
        attribute = attrs.fields_dict(cls)[attr_name]
        metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, {})
        name = metadata.get(cls.NAME_KEY, attr_name)
        path = metadata.get(cls.ROOT_KEY, cls.get_root_hdf5_path(prefix=prefix))
        return path, name


@attrs.define(repr=True, eq=True)
class Resourced(metaclass=abc.ABCMeta):
    RESOURCE: ClassVar[Resource] = Resource.NONE()

    @classmethod
    def from_resources(cls, resources: dict[Resource, Any]) -> Resourced:
        if cls.RESOURCE in resources:
            return resources[cls.RESOURCE]

        args = []
        kwargs = {}

        attrs.resolve_types(cls)
        for attr_name, attribute in attrs.fields_dict(cls).items():
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, {})
            if build_func := getattr(cls, f'{attr_name}_from_resources', None):
                value = build_func(resources)

            elif isinstance(attribute.type, type) and issubclass(attribute.type, Consolidator):
                value = resources[attribute.type.RESOURCE]

            elif is_union(attribute.type) and (bases := get_union_subclasses(attribute.type, Consolidator)):
                # TODO: apply each base to until first successful one
                value = resources.get(bases[0].RESOURCE)

            elif resource := metadata.get('resource'):
                console.log(attr_name, resource)
                value = resources.get(resource)

            else:
                continue

            if attribute.kw_only or attribute.default is not attrs.NOTHING:
                kwargs[attr_name] = value
            else:
                args.append(value)
        return cls(*args, **kwargs)

    @classmethod
    def get_build_resources(cls) -> Resource:
        resources = Resource.NONE()
        for attr_name, attribute in attrs.fields_dict(cls).items():
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, {})
            if resc_func := getattr(cls, f'get_{attr_name}_build_resources', None):
                resources |= resc_func()
            elif attribute.default is not attrs.NOTHING or attribute.init:
                resources |= metadata.get('resource', Resource.NONE())

        return Resource.NONE()


@attrs.mutable(repr=True, eq=True)
class Consolidator(Resourced, Archived, metaclass=abc.ABCMeta):
    RESOURCE_IDX_COL: ClassVar[str] = 'resource_idx'

    df: pd.DataFrame = attrs.field(
        validator=attrs.validators.instance_of(pd.DataFrame),
        repr=False
    )

    @classmethod
    @abc.abstractmethod
    def df_from_resources(cls, resources: dict[Resource, Any]) -> pd.DataFrame:
        raise NotImplementedError()

    def df_to_hdf5(self, filename: pathlib.Path, /, prefix: str = '') -> None:
        path, name = self.get_attr_hdf5_path('df', prefix=prefix)
        with pd.HDFStore(str(filename), mode='a') as hdf5store:
            self.df.to_hdf(
                path_or_buf=hdf5store,
                key='/'.join([path, name]),
                format='table'
            )

    @classmethod
    def df_from_hdf5(cls, filename: pathlib.Path, /, prefix: str = '') -> pd.DataFrame:
        path, name = cls.get_attr_hdf5_path('df', prefix=prefix)
        with pd.HDFStore(str(filename), mode='r') as hdf5store:
            df = pd.read_hdf(path_or_buf=hdf5store, key='/'.join([path, name]))
        return cast(pd.DataFrame, df)

    # def analyze(self, *analyzers: Sequence[Analyzer[Self]]) -> None:
    #     raise NotImplementedError

    def __len__(self) -> int:
        return len(self.df)
