import pathlib

import numpy as np
import pytest

from mctools.core import Resources

from mctools.parsing import RouteParser, LogParser
from mctools.parsing.core.error import ParsingError, InvalidFile



calculation_name = 'rasci_1r1h36o_2r13e14o_3r1p6o.casscf_13e14o_ref.ano-yb_r1-i_r0.ybi6_1.gdv_j14p.16262576'
data_dir = pathlib.Path().resolve() / 'data'
log = data_dir / f'{calculation_name}.log'
fchk = log.with_suffix('.fchk')


class TestRouteParser:
    def test_prepared(self):
        parser = RouteParser()
        with pytest.raises(InvalidFile):
            route, *_ = parser.parse_file(None)

    def test_no_route(self):
        parser = RouteParser()
        with pytest.raises(ParsingError):
            route, *_ = parser.parse(fchk)

    def test_route_runs_parser(self):
        parser = RouteParser()
        route, *_ = parser.parse(log)
        assert route.is_complete


class TestLinksParser:
    pass


class TestLogParser:
    def test_valid_log(self):
        parser = LogParser(resources=Resources.ao_int1e_stv)
        route, data = parser.parse(log)

        assert route.is_complete
        assert len(data) == len(Resources.ao_int1e_stv)
        for resource in Resources.ao_int1e_stv:
            assert resource in data

            matrix = data[resource]
            assert isinstance(matrix, np.ndarray)
            assert matrix.ndim == 2

            rows, colums = matrix.shape
            assert rows == colums

            assert np.array_equal(matrix.T, matrix)
