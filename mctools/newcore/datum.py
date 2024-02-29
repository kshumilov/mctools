from __future__ import annotations

import abc
from typing import Any

import attrs
import numpy as np
import pandas as pd
from numpy import typing as npt

from .resource import Resource
from .consolidator import Consolidator, Storage
from .metadata import DatumFieldMetadata, MCTOOLS_METADATA_KEY, ConsolidatorFieldMetadata


class ResourceError(Exception):
    pass


class DependencyNotFound(ResourceError):
    pass


class DimensionMismatch(ResourceError):
    pass


@attrs.define(repr=True, eq=True)
class Datum(metaclass=abc.ABCMeta):
    @classmethod
    def get_resources(cls) -> dict[Resource, DatumFieldMetadata]:
        resources: dict[Resource, DatumFieldMetadata] = {}
        for attribute in cls.__attrs_attrs__:
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, DatumFieldMetadata())
            if metadata.is_resourceful:
                resources[metadata.res_spec.label] = metadata
        return resources

    @classmethod
    def get_tabular_build_resources(cls) -> dict[Resource, DatumFieldMetadata]:
        df_cols: dict[Resource, DatumFieldMetadata] = {}
        for attribute in cls.__attrs_attrs__:
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, DatumFieldMetadata())
            if metadata.required and metadata.is_tabular and metadata.is_resourceful:
                df_cols[metadata.res_spec.label] = metadata
        return df_cols

    @classmethod
    def get_nontabular_indexed_build_resources(cls) -> dict[Resource, DatumFieldMetadata]:
        resources: dict[Resource, DatumFieldMetadata] = {}
        for attribute in cls.__attrs_attrs__:
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, DatumFieldMetadata())
            if metadata.required and not metadata.is_tabular and metadata.is_resourceful:
                resources[metadata.res_spec.label] = metadata
        return resources

    @classmethod
    def get_index_label(cls) -> str:
        return f'{cls.__name__.lower()}_idx'


class DataConsolidator(Consolidator):
    @classmethod
    def from_resources(cls, storage: Storage) -> Consolidator:
        df = cls.build_df(storage)
        nonindexed = cls.get_non_indexed_resources(storage)
        indexed = cls.get_indexed_non_df_resources(storage, len(df))
        return cls(df, *nonindexed, **indexed)

    @classmethod
    def build_df(cls, storage: Storage) -> pd.DataFrame:
        df_dict: dict[str, npt.ArrayLike] = {}

        n_datapoints: int | None = None
        df_cols = cls.DatumClass.get_tabular_build_resources()
        for resource_label, metadata in df_cols.items():
            if (resource := storage.get(resource_label, None)) is None:
                raise DependencyNotFound(f"Required resource {metadata.res_spec.label} is missing")

            if resource.ndim > 2:
                raise DimensionMismatch(f"Resource is of wrong dimension, "
                                        f"cannot be more than 2: ndim = {resource.ndim}")

            if resource.ndim == 1:
                n_rows = resource.shape[0]
                n_cols = 1
            else:
                n_rows = resource.shape[metadata.df_spec.idx_axis]
                n_cols = resource.shape[metadata.df_spec.value_axis]

            if n_cols != metadata.df_spec.n_cols:
                raise DimensionMismatch(f"Resource's value axis does not match "
                                        f"number of columns: "
                                        f"{n_cols} != {metadata.df_spec.n_cols}")

            if n_datapoints is not None and n_rows != n_datapoints:
                raise DimensionMismatch(f"Resource's index direction does not match "
                                        f"previously identified number of datapoints: "
                                        f"{n_datapoints} != {n_rows}")
            else:
                n_datapoints = n_rows

            # Orient resource such that it is in "long-table" format
            if resource.ndim == 1:
                df_dict[metadata.df_spec.cols[0]] = resource
            else:
                resource = resource.T if metadata.df_spec.idx_axis == 0 else resource
                for col_name, col in zip(metadata.df_spec.cols, resource):
                    df_dict[col_name] = col

            df = pd.DataFrame.from_dict(df_dict)
            df[cls.RESOURCE_IDX_COL] = np.arange(len(df))
            df.index.name = cls.DatumClass().get_index_label()
            return df

    @classmethod
    def get_indexed_non_df_resources(cls, storage: Storage, n_datapoints: int) -> dict[str, Any]:
        required_resources: dict[str, Any] = {}
        for resource_label, metadata in cls.DatumClass.get_nontabular_indexed_build_resources().items():
            if (resource := storage.get(resource_label, None)) is None:
                raise DependencyNotFound(f"Required resource "
                                         f"{metadata.res_spec.label} is missing")

            n_rows = resource.shape[metadata.res_spec.idx_axis]
            if n_datapoints != n_rows:
                raise DimensionMismatch(f"Resource's index direction does not match "
                                        f"previously identified number of datapoints: "
                                        f"{n_datapoints} != {n_rows}")

            required_resources[metadata.res_spec.label.name] = resource

        return required_resources

    @classmethod
    def get_non_indexed_resources(cls, storage: Storage) -> tuple[Resource, ...]:
        resources: list[Resource] = []
        for attribute in cls.__attrs_attrs__:
            metadata = attribute.metadata.get(MCTOOLS_METADATA_KEY, ConsolidatorFieldMetadata())
            if metadata.is_resourceful and not metadata.res_spec.is_indexed:
                resources[metadata.res_spec.label.name] = storage[metadata.res_spec.label]

        return tuple(resources)
