from __future__ import annotations

import abc

from typing import TypeVar

from core.resources import Resources

from parsing.core.error import ParsingError
from parsing.core.stepper import Predicate
from parsing.core.parser import Listener, TaskFileParser, ListenerSettings

from .stepper import LogStepper
from ..route import Link, IOps

__all__ = [
    'LinkParser', 'LinkParserType',
]


class LinkParser(TaskFileParser, metaclass=abc.ABCMeta):
    DefaultStepper = LogStepper

    LINK_ANCHOR_TEMPLATE: str = '%s.exe'
    TERMINATION_ANCHOR: str = 'Leave Link'

    def __init__(self, iops: IOps, /, **kwargs) -> None:
        # Before super, because used by ResourceHandler.get_resources(),
        self.iops = iops
        super(LinkParser, self).__init__(**kwargs)

    @property
    def link(self) -> Link:
        name = self.__class__.__name__[:-len('Parser')]
        return Link[name]

    @property
    def anchor(self) -> str:
        return self.LINK_ANCHOR_TEMPLATE % self.link.value

    def get_resources(self, requested_resources: Resources, /) -> Resources:
        requested_resources = super(LinkParser, self).get_resources(requested_resources)
        return self.parse_iops(requested_resources)

    @abc.abstractmethod
    def parse_iops(self, requested_resources: Resources, /) -> Resources:
        return self.DEFAULT_RESOURCES

    def execute_dispatch(self) -> bool:
        print(f'{self!r}: Dispatching')
        until = self.stepper.get_predicate(self.TERMINATION_ANCHOR)
        if not self.stepper.step_to_first(self.dispatch, until) and self.total_dispatches == 0:
            raise ParsingError(f'{self!r}: Did not find any anchors')
        print(f'{self!r}: Done dispatching')
        return True

    def get_link_predicate(self) -> Predicate:
        return self.stepper.get_predicate(self.anchor)

    def get_link_listener(self) -> Listener:
        return Listener(
            label=self.link,
            anchor=self.anchor,
            predicate=self.get_link_predicate(),
            handle=self.parse,
            settings=ListenerSettings(
                dispatch_file=True,
            )
        )


LinkParserType = TypeVar('LinkParserType', bound=LinkParser, covariant=True)
