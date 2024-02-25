import pathlib

import io

import h5py
import numpy as np
import pytest
from icecream import ic

from core.resource import Resource
from parsing.core.error import ParsingError

from parsing.gaussian.log.route import RouteParser, Route
from parsing.gaussian.log.parser import LogParser
from parsing.gaussian.log.links import LinksParser, L910Parser
from parsing.gaussian.log.links.l302 import L302Parser
from parsing.gaussian.log.route.route import Link


@pytest.fixture
def route_parser():
    return RouteParser()


class TestRouteParser:
    def test_route_parser_parses_log(self, route_parser: RouteParser, logfile: io.TextIOWrapper):
        r, file = route_parser.parse(logfile)
        assert r.is_complete

    def test_no_route(self, route_parser: RouteParser, fchkfile: io.TextIOWrapper):
        with pytest.raises(ParsingError):
            route_parser.parse(fchkfile)


@pytest.fixture
def route(logname: pathlib.Path, route_parser: RouteParser) -> Route:
    r, *_ = route_parser.parse(logname)
    return r


class TestLinksParser:
    def test_init(self):
        parser = LinksParser()
        assert Link.L302 in parser.DEFAULT_LINK_PARSER_CLASSES
        assert not parser.route.is_complete

    def test_build_parsers_empty(self):
        parser = LinksParser()
        parser.build_parsers()
        assert len(parser) == 0

    def test_build_parsers_with_route(self, route: Route):
        parser = LinksParser(route=route)
        parser.build_parsers()
        assert len(parser) == 2

        p1 = parser.link_parsers.parsers[0]
        assert isinstance(p1, L302Parser)

        p2 = parser.link_parsers.parsers[1]
        assert isinstance(p2, L910Parser)

    def test_parses_with_route(self, route: Route, logfile):
        parser = LinksParser(route=route)
        data, logfile = parser.parse(logfile)

        rdms = data[Resource.ci_one_rdms]
        osc = data[Resource.ci_osc]

        assert route.is_complete


class TestLogParser:
    def test_valid_log(self, logname):
        parser = LogParser()
        (route, data), file = parser.parse(logname)

        filename = '/Users/kirill/projects/mctools/tests/data/casscf/gdv.h5'
        with h5py.File(filename, 'w', libver='latest') as f:
            for label, resource in data.items():
                if isinstance(resource, np.ndarray):
                    name = '/'.join(label.name.split('_'))
                    f.create_dataset(
                        name, data=resource, dtype=resource.dtype,
                        compression='gzip'
                    )

        assert route.is_complete

        # assert len(data) == len(Resources.ao_int1e_stv)
        # for resource in Resources.ao_int1e_stv:
        #     assert resource in data
        #
        #     matrix = data[resource]
        #     assert isinstance(matrix, np.ndarray)
        #     assert matrix.ndim == 2
        #
        #     rows, colums = matrix.shape
        #     assert rows == colums
        #
        #     assert np.array_equal(matrix.T, matrix)
