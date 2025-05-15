import logging
from pythonjsonlogger import jsonlogger


def setup_logging(level: int = logging.INFO):
    """
    Configure root logger to emit structured JSON logs.
    """
    handler = logging.StreamHandler()
    fmt = jsonlogger.JsonFormatter(
        "%(asctime) %(levelname) %(name) %(message)"
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    root.handlers = []      # remove default handlers
    root.addHandler(handler)
    root.setLevel(level)