from __future__ import annotations

import attrs

__all__ = [
    'ParsingError',
    'EOFReached',
    'TerminatorReached',
    'ParserNotPrepared',
    'AnchorNotFound',
]


@attrs.define(init=False)
class ParsingError(Exception):
    def __init__(self, *args, **kwargs) -> None: # real signature unknown
        pass


@attrs.define(init=False)
class TerminatorReached(ParsingError):
    pass


@attrs.define(init=False)
class ParserNotPrepared(ParsingError):
    pass


@attrs.define(init=False)
class EOFReached(ParsingError):
    pass


@attrs.define(init=False)
class AnchorNotFound(ParsingError):
    pass
