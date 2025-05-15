# File: parser/registry.py

import pkgutil
import importlib
from pathlib import Path
from typing import List

from parser.base import Parser


def load_parsers() -> List[Parser]:
    """
    Discover and instantiate all Parser subclasses in parser/ directory.
    Order is determined by filesystem order (LLMParser first, then RegexFallbackParser).
    """
    parsers: List[Parser] = []
    pkg_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(pkg_path)]):
        module = importlib.import_module(f"parser.{module_info.name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, Parser)
                and attr is not Parser
            ):
                parsers.append(attr())
    return parsers