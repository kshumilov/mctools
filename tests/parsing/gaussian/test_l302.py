import pytest
import numpy as np

from parsing.core import DispatchParser
from parsing.gaussian.log.links.base import RealMatrixParser


class TestInt1eParsing:
    def test_parse_overlap(self, logname):
        target = 'Overlap'
        shape = (575, 575)

        l302parser = DispatchParser()
        l302parser.build_listener_from_parser('Overlap', RealMatrixParser(), max_runs=1)
        data = l302parser.parse(logname)

        assert target in data
        array = data[target][0]
        assert isinstance(array, np.ndarray)
        assert array.shape == shape
        assert np.array_equal(array, array.T)

    def test_parse_stv(self, logname):
        shape = (575, 575)

        l302parser = DispatchParser()

        targets = ['Overlap', 'Kinetic', 'Core Hamiltonian']
        for target in targets:
            l302parser.build_listener_from_parser(target, RealMatrixParser(), max_runs=1)

        data = l302parser.parse(logname)

        for target in targets:
            assert target in data
            matrix = data[target][0]

            assert isinstance(matrix, np.ndarray)
            assert matrix.shape == shape
            assert np.array_equal(matrix, matrix.T)
