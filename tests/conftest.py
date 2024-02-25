from __future__ import annotations

from typing import TYPE_CHECKING


import pytest


if TYPE_CHECKING:
    from io import TextIOWrapper
    from pathlib import Path


@pytest.fixture(params=['gaussian/casscf'])
def calc(request, shared_datadir: Path) -> Path:
    return shared_datadir / request.param


@pytest.fixture
def logname(calc: Path) -> Path:
    return calc / f'gdv.log'


@pytest.fixture
def fchkname(calc: Path) -> Path:
    return calc / f'gdv.fchk'


@pytest.fixture
def fchkfile(fchkname: Path) -> TextIOWrapper:
    with fchkname.open('r') as f:
        yield f


@pytest.fixture
def logfile(logname: Path) -> TextIOWrapper:
    with logname.open('r') as f:
        yield f


@pytest.fixture
def top3log(logname: Path) -> list[str]:
    with open(logname, 'r', encoding='utf') as f:
        lines = [line for line, _ in zip(f, range(3))]
    return lines
