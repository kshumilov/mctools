from io import IOBase
from typing import IO, AnyStr

__all__ = [
    'short_file_repr',
    'check_file',
]

MAX_LENGTH = 30


def shorten_filename(filename: str, /, max_length: int = MAX_LENGTH,  sep='...') -> str:
    if len(filename) > (max_length - len(sep)):
        left_length = max_length // 2
        right_length = max_length - left_length
        return filename[:left_length] + sep + filename[-right_length:]
    return filename


def short_file_repr(file: IO[AnyStr], /, **kwargs) -> str:
    if file:
        return shorten_filename(file.name, **kwargs)
    return str(None)


def check_file(file: IO[AnyStr]) -> bool:
    return isinstance(file, IOBase) and not file.closed and file.readable() and file.seekable()
