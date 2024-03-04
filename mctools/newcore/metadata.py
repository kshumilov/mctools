from __future__ import annotations

from typing import Callable, Any

import attrs

from mctools.newcore.resource import Resource


__all__ = [
    'MCTOOLS_METADATA_KEY',
    # 'DFFieldMetadata',
    # 'ResourcedFieldMetadata',
    # 'DatumFieldMetadata',
    # 'ConsolidatorFieldMetadata',
]


MCTOOLS_METADATA_KEY = '__mctools_metadata'


@attrs.define(repr=True, eq=True)
class Meta:
    archive_root: str = ''
    archive_name: str = ''
    archive_ignore: bool = False
    to_archive: Callable[[Any], Any] = None
    resource: Resource | None = None



@attrs.frozen(repr=True, eq=True, hash=True)
class DFFieldMetadata:
    # How to store data on the df
    idx_axis: int = 0
    value_axis: int = 1
    cols: tuple[str, ...] = attrs.field(
        factory=tuple,
        validator=attrs.validators.min_len(1),
    )

    @property
    def n_cols(self) -> int:
        return len(self.cols)


@attrs.frozen(repr=True, eq=True, hash=True)
class ResourcedFieldMetadata:
    # How to map resource to df
    label: Resource
    indexed: bool = True
    # TODO: add convenient name for the resource on Consolidator class


@attrs.frozen(repr=True, eq=True, hash=True)
class DatumFieldMetadata:
    required: bool = False
    df_spec: DFFieldMetadata | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(DFFieldMetadata)
        )
    )
    res_spec: ResourcedFieldMetadata | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(ResourcedFieldMetadata)
        )
    )

    def __attrs_post_init__(self) -> None:
        if self.df_spec is None and self.res_spec is None:
            raise ValueError('Either df_spec or res_spec must be set')

    @property
    def is_resourceful(self) -> bool:
        return self.res_spec is not None

    @property
    def is_tabular(self) -> bool:
        return self.df_spec is not None

    @property
    def is_indexed(self) -> bool:
        return self.df_spec is not None or self.res_spec.indexed


@attrs.frozen(repr=True, eq=True, hash=True)
class ConsolidatorFieldMetadata:
    required: bool = False
    res_spec: ResourcedFieldMetadata | None = attrs.field(
        default=None,
        validator=attrs.validators.optional(
            attrs.validators.instance_of(ResourcedFieldMetadata)
        )
    )

    @property
    def is_resourceful(self) -> bool:
        return self.res_spec is not None
