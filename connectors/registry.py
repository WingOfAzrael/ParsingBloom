# File: connectors/registry.py

import pkgutil
import importlib
from typing import Dict, Any
from connectors.base import Connector


def load_connector(name: str, cfg: Dict[str, Any]) -> Connector:
    """
    Dynamically load and instantiate the connector class named `name`
    from `connectors/{name}.py`, passing it its config dict.
    """
    module = importlib.import_module(f"connectors.{name}")
    for attr in dir(module):
        cls = getattr(module, attr)
        if (isinstance(cls, type)
            and issubclass(cls, Connector)
            and cls is not Connector
        ):
            return cls(cfg)
    raise ValueError(f"No connector plugin found for '{name}'")
