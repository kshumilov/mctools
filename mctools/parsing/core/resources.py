from __future__ import annotations

from typing import TypeVar, ClassVar

from core import Resources

__all__ = [
    'ResourceHandler',
]


class ResourceHandler(object):
    """A base class for handling resources.

    Attributes:
        resources: Resources defined on this instance
    """

    DEFAULT_RESOURCES: ClassVar[Resources] = Resources.NONE

    resources: Resources | None = None

    def __init__(self, /, requested_resources: Resources | None = None, **kwargs) -> None:
        print('Initializing:', self.__class__.__name__, 'ResourceHandler', **kwargs)
        requested_resources = requested_resources or self.DEFAULT_RESOURCES
        self.resources = self.get_resources(requested_resources)

    def get_resources(self, requested_resources: Resources, /) -> Resources:
        """Filters out resources that are cannot be processed by this handler"""
        self.resources = requested_resources
        return requested_resources & self.DEFAULT_RESOURCES


ResourceParserType = TypeVar('ResourceParserType', bound=ResourceHandler, covariant=True)
