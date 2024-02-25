from __future__ import annotations

from parsing.gaussian.log.links.parser import LinksParser
from parsing.gaussian.log.links.l302 import L302Parser
from parsing.gaussian.log.links.l910 import L910Parser

__all__ = [
    'LinksParser',
]

LinksParser.register(L302Parser.get_link(), L302Parser)
LinksParser.register(L910Parser.get_link(), L910Parser)

