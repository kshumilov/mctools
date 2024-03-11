import io

import pytest

from mctools.parsing.core.error import ParsingError

from mctools.parsing.gaussian.log.route import RouteParser
from mctools.parsing.gaussian.log.parser import LogParser
from mctools.parsing.gaussian.fchk import FchkParser


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


class TestLogParser:
    def test_valid_log(self, logfile: io.TextIOWrapper):
        parser = LogParser()
        (route, data), file = parser.parse(logfile)
        assert route.is_complete


class TestFchkParser:
    def test_valid_fchk(self, fchkfile: io.TextIOWrapper):
        parser = FchkParser()
        data, file = parser.parse(fchkfile)

        assert isinstance(data, dict)
