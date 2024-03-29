from __future__ import annotations

from typing import TYPE_CHECKING


import pytest


if TYPE_CHECKING:
    from io import TextIOWrapper
    from pathlib import Path


@pytest.fixture(params=['gaussian/kedge'])
def calc(request, shared_datadir: Path) -> Path:
    return shared_datadir / request.param


@pytest.fixture
def logname(calc: Path) -> Path:
    return calc / 'rasci-1r1h12o_2r49e50o_3r1p10o.casscf_sa50_50s_ref.tzvpall.ybi6.bl_315.gdv_j14p.16645549.log'


@pytest.fixture
def fchkname(calc: Path) -> Path:
    return calc / 'rasci-1r1h12o_2r49e50o_3r1p10o.casscf_sa50_50s_ref.tzvpall.ybi6.bl_315.gdv_j14p.16645549.fchk'


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
