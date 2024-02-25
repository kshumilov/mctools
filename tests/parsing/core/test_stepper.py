import io
from typing import TypeAlias, AnyStr, Any

import pytest
from icecream import ic

from parsing.core.error import ParsingError, EOFReached
from parsing.core.stepper import LineStepper
from parsing.core.filehandler import FileWithPosition, FileHandler, DoesNotHaveAFile
from parsing.core.utils import Anchor, get_str_predicate, Predicate


FH: TypeAlias = FileHandler[AnyStr]
FWP: TypeAlias = FileWithPosition[AnyStr]


@pytest.fixture(params=[FileHandler, LineStepper])
def empty_handler(request) -> FileHandler[Any]:
    yield request.param()


@pytest.fixture
def fchk(fchkfile: io.TextIOWrapper) -> FWP[str]:
    return FileWithPosition(fchkfile)


@pytest.fixture
def handler(empty_handler: FH[Any], fchk: FWP[str]) -> FH[str]:
    empty_handler.take(fchk)
    return empty_handler


class TestFileHandler:
    def test_init(self, empty_handler: FileHandler):
        assert empty_handler.fwp is None

    def test_empty_raises_on_return_file(self, empty_handler):
        with pytest.raises(DoesNotHaveAFile):
            empty_handler.return_file()

    def test_raises_on_take_with_invalid_file(self, empty_handler: FileHandler, fchk: FileWithPosition):
        fchk.file.close()
        with pytest.raises(ValueError):
            empty_handler.take(fchk)

    def test_empty_take_valid_file(self, empty_handler: FileHandler, fchk: FileWithPosition):
        empty_handler.take(fchk)
        assert empty_handler.fwp is fchk

    def test_return_file(self, handler: FileHandler):
        orig_file = handler.fwp
        fwp = handler.return_file()
        assert orig_file is fwp


@pytest.fixture
def empty_linestepper() -> LineStepper:
    yield LineStepper()


@pytest.fixture
def linestepper(logfile) -> LineStepper:
    yield LineStepper.from_file(logfile)


@pytest.fixture
def overlap_header_predicate() -> Predicate:
    return LineStepper.get_anchor_predicate('Overlap')


@pytest.fixture
def enter_link_predicate() -> Predicate:
    return LineStepper.get_anchor_predicate('(Enter')


@pytest.fixture
def fake_predicate() -> Predicate:
    return LineStepper.get_anchor_predicate('blkasdihf')


class TestLineStepper:
    def test_init_state(self, empty_linestepper: LineStepper):
        assert empty_linestepper.fwp is None

    def test_empty_step_to_raises(self, empty_linestepper: LineStepper):
        with pytest.raises(DoesNotHaveAFile):
            empty_linestepper.step_to(lambda l: 'Entering ' in l)

    def test_empty_step_until_raises(self, empty_linestepper: LineStepper):
        with pytest.raises(DoesNotHaveAFile):
            empty_linestepper.step_until(
                lambda l: 'Entering ' in l,
                lambda l: 'Done' in l,
            )

    def test_readline(self, linestepper: LineStepper, top3log: list[str]):
        assert linestepper.readline() == top3log[0]

    def test_step_forward(self, linestepper: LineStepper, top3log: list[str]):
        for i, topline in enumerate(top3log):
            assert linestepper.step()
            assert linestepper.fwp.last_line == topline
            assert linestepper.fwp.n_lines_read == i + 1

    def test_step_back_to_beginning(self, linestepper: LineStepper):
        linestepper.step()
        assert not linestepper.step_back()
        assert linestepper.fwp.last_line == ''

    def test_step_back_middle(self, linestepper: LineStepper):
        linestepper.step()
        linestepper.step()
        assert linestepper.step_back()

    def test_step_to_fresh(self, linestepper: LineStepper, overlap_header_predicate: Predicate):
        assert linestepper.step_to(overlap_header_predicate)
        assert linestepper.fwp.last_line == ' *** Overlap *** \n'
        assert linestepper.fwp.n_lines_read == 794

    def test_step_to_from_previous_read(self, linestepper: LineStepper, overlap_header_predicate: Predicate):
        linestepper.step_to(overlap_header_predicate)
        assert linestepper.step_to(overlap_header_predicate, check_last_read=True)
        assert linestepper.fwp.last_line == ' *** Overlap *** \n'
        assert linestepper.fwp.n_lines_read == 794

    def test_step_to_not_found_skip(self, linestepper: LineStepper, fake_predicate: Predicate):
        assert not linestepper.step_to(fake_predicate)

    def test_step_to_not_found_raise(self, linestepper: LineStepper, fake_predicate: Predicate):
        with pytest.raises(EOFReached):
            linestepper.step_to(fake_predicate, on_eof='raise')

    def test_step_until(self, linestepper, overlap_header_predicate: Predicate, enter_link_predicate: Predicate):
        assert not linestepper.step_until(overlap_header_predicate, enter_link_predicate)
        assert linestepper.fwp.last_line == ' (Enter /sw/contrib/chem-src/gdv/j14p/gdv/l101.exe)\n'
        assert linestepper.fwp.n_lines_read == 5120

