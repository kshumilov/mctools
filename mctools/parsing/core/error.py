from __future__ import annotations


__all__ = [
    'ParsingError',
    'InvalidFile',
    'EOFReached',
    'ParserNotPrepared',
]


class ParsingError(Exception):
    pass


class InvalidFile(ParsingError):
    pass


class ParserNotPrepared(ParsingError):
    pass


class EOFReached(ParsingError):
    pass
