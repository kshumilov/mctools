import pathlib

import pytest

from mctools.parsing.core import LineStepper
from parsing.core.error import ParsingError, EOFReached
from parsing.core.stepper import FileHandler
from parsing.core.utils import shorten_filename

calculation_name = 'rasci_1r1h36o_2r13e14o_3r1p6o.casscf_13e14o_ref.ano-yb_r1-i_r0.ybi6_1.gdv_j14p.16262576'

data_dir = pathlib.Path().resolve() / 'data'
log = data_dir / f'{calculation_name}.log'
fchk = log.with_suffix('.fchk')


class TestFileHandler:
    def test_init(self):
        handler = FileHandler()
        assert handler.file is None
        assert not handler.workable
        assert repr(handler) == f'{FileHandler.__name__}(file=None)'

    def test_set_file(self):
        handler = FileHandler()

        with pytest.raises(ValueError):
            handler.set_file(None)

        handler.set_file(None, check=False)
        assert handler.file is None

        shortname = shorten_filename(str(log))
        with log.open('r') as f:
            handler.set_file(f)
            assert handler.file is f
            assert handler.workable
            assert repr(handler) == f'{FileHandler.__name__}(file={shortname})'

        assert not handler.workable

    def test_unset_file(self):
        handler = FileHandler()

        with log.open('r') as f:
            handler.set_file(f)

            result = handler.unset_file()
            assert result is f
            assert not handler.workable

        assert not handler.workable

    def test_from_file(self):
        with log.open('r') as f:
            handler = FileHandler.from_file(f)
            assert handler.file is f
            assert handler.workable

        with pytest.raises(ValueError):
            FileHandler.from_file(f)


class TestLineStepper:
    def test_init(self):
        stepper = LineStepper()
        assert stepper.file is None
        assert stepper.line is None
        assert not stepper.workable
        assert stepper.line_offset is None
        assert not stepper.is_running
        assert repr(stepper) == f"{LineStepper.__name__}(file=None, on_eof='skip')"

    def test_set_file(self):
        stepper = LineStepper()

        with pytest.raises(Exception):
            stepper.set_file(None)

        stepper.set_file(None, check=False)
        assert stepper.file is None
        assert stepper.line is None
        assert not stepper.workable
        assert stepper.line_offset is None
        assert not stepper.is_running
        assert repr(stepper) == f"{LineStepper.__name__}(file=None, on_eof='skip')"

        shortname = shorten_filename(str(log))
        with log.open('r') as f:
            stepper.set_file(f)
            assert stepper.file is f
            assert stepper.line is None
            assert stepper.workable
            assert stepper.line_offset is None
            assert not stepper.is_running
            assert repr(stepper) == f"{LineStepper.__name__}(line=None, file={shortname}, on_eof='skip')"

        assert not stepper.workable

    def test_basic_steps(self):
        shortname = shorten_filename(str(log))

        first_line = ' Entering Gaussian System, Link 0=gdv\n'
        second_line = ' Input=rasci_1r1h36o_2r13e14o_3r1p6o.casscf_13e14o_ref.ano-yb_r1-i_r0.ybi6_1.com\n'
        third_line = ' Output=rasci_1r1h36o_2r13e14o_3r1p6o.casscf_13e14o_ref.ano-yb_r1-i_r0.ybi6_1.log\n'
        with log.open('r') as f:
            stepper = LineStepper.from_file(f)

            # First Step
            assert stepper.readline() == first_line
            assert stepper.line == first_line
            assert stepper.workable
            assert stepper.is_running
            assert stepper.line_offset == 0
            assert repr(stepper) == f"{LineStepper.__name__}(line={first_line[:10]!r}, line_offset=0, file={shortname}, on_eof='skip')"

            # Second Step
            assert stepper.step_forward()
            assert stepper.line == second_line
            assert stepper.workable
            assert stepper.is_running
            assert stepper.line_offset == len(first_line)

            # Third Step
            assert stepper.step_forward()
            assert stepper.line == third_line
            assert stepper.is_running
            assert stepper.line_offset == len(first_line) + len(second_line)

            # First Back
            assert stepper.step_back()
            assert stepper.line is None
            assert stepper.line_offset is None

    def test_step_back(self):
        with pytest.raises(ParsingError):
            with log.open('r') as f:
                stepper = LineStepper.from_file(f)
                stepper.step_back()

    def test_step_to(self):
        with log.open('r') as f:
            stepper = LineStepper.from_file(f)
            predicate = stepper.get_str_predicate('Overlap')

            assert stepper.step_to(predicate)
            assert stepper.line == ' *** Overlap *** \n'
            assert repr(stepper) == "LineStepper(line=' *** Overl', line_offset=175783, file=/Users/kirill/P...4p.16262576.log, on_eof='skip')"
            assert stepper.line_offset == 175783

            assert stepper.step_to(predicate, check_last_read=True)
            assert stepper.line == ' *** Overlap *** \n'
            assert stepper.line_offset == 175783

        with fchk.open('r') as f:
            stepper = LineStepper.from_file(f)
            predicate = stepper.get_str_predicate('blkasdihf')
            assert not stepper.step_to(predicate)

    def test_raises_eof(self):
        with pytest.raises(EOFReached):
            f = fchk.open('r')
            stepper = LineStepper.from_file(f)
            stepper.on_eof = 'raise'
            predicate = stepper.get_str_predicate('blkasdihf')
            stepper.step_to(predicate)

    def test_step_until(self):
        with log.open('r') as f:
            stepper = LineStepper.from_file(f)
            predicate = stepper.get_str_predicate('Overlap')
            until = stepper.get_str_predicate('(Enter')

            assert not stepper.step_to_first(predicate, until)
            assert stepper.line == ' (Enter /sw/contrib/chem-src/gdv/j14p/gdv/l101.exe)\n'
            assert stepper.line_offset == 5322

    def test_unset_file(self):
        with log.open('r') as f:
            stepper = LineStepper.from_file(f)

            result = stepper.unset_file()
            assert result is f
            assert stepper.file is None
            assert not stepper.workable

        assert not stepper.workable


