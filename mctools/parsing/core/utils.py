from __future__ import annotations

from typing import IO, AnyStr

__all__ = [
    'short_file_repr',
]

MAX_LENGTH = 30


def shorten_filename(filename: str, /, max_length: int = MAX_LENGTH, sep: str = '...') -> str:
    if len(filename) > (max_length - len(sep)):
        left_length = max_length // 2
        right_length = max_length - left_length
        return filename[:left_length] + sep + filename[-right_length:]
    return filename


def short_file_repr(file: IO[AnyStr], /, max_length: int = MAX_LENGTH, sep: str = '...') -> str:
    if file:
        return shorten_filename(file.name, max_length=max_length, sep=sep)
    return str(None)



# PKey: TypeAlias = type[Parser] | tuple[type[Parser], int]
#
#
# @attrs.define(repr=True, eq=True, frozen=True, hash=True)
# class ParserKey:
#     parser_class: type[Parser] = attrs.field(
#         validator=attrs.validators.instance_of((type(Parser),))
#     )
#     index: int = attrs.field(default=0, validator=attrs.validators.ge(0))
#
#     @classmethod
#     def convert_key(cls, key: ParserKey | type[Parser] | tuple[type[Parser], int], /) -> ParserKey:
#         match key:
#             case cls():
#                 return key
#             case (parser_class, index) if issubclass(parser_class, Parser):
#                 return cls(parser_class, index)
#             case parser_class if issubclass(parser_class, Parser):
#                 return cls(parser_class)
#             case _:
#                 raise KeyError(f"Invalid parser_key: {key!r}")
#
#     @classmethod
#     def from_parser(cls, parser: Parser, /, index: int = 0) -> ParserKey:
#         return cls(parser.__class__, index)
