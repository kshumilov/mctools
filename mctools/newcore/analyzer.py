from __future__ import annotations

from typing import Protocol, Any, TypeVar

from .consolidator import Consolidator

C = TypeVar('C', bound=Consolidator)


class Analyzer(Protocol[C]):
    def __cal__(self, consolidator: C) -> Any:
        ...
