import os, yaml
from pathlib import Path
from typing import IO, Union

def load_config(src: Union[str, Path, IO[str]] = "config/config.yaml") -> dict:
    """
    Parameters
    ----------
    src : str | pathlib.Path | IO[str]
        • If *str* or *Path*, the file is opened inside this function.  
        • If an *IO* object, the caller remains responsible for closing it.

    Returns
    -------
    dict
        Parsed YAML with any PARSINGFORGE_DEVICE overrides applied.
    """
    # --------------------------------------------------------------------- #
    # 1) Read YAML safely                                                   #
    # --------------------------------------------------------------------- #
    if isinstance(src, (str, Path)):
        with open(src, "r", encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
    else:  # assume file-like object
        cfg = yaml.safe_load(src) or {}

    # Guarantee the nested dict exists before we mutate it
    cfg.setdefault("parser", {})

    # --------------------------------------------------------------------- #
    # 2) Environment override (unchanged behaviour)                         #
    # --------------------------------------------------------------------- #
    override = os.getenv("PARSINGFORGE_DEVICE")  # "cpu" | "cuda" | None
    if override:
        if override.lower() == "cpu":
            cfg["parser"]["device"] = -1          # HF & llama-cpp
            cfg["parser"]["device_map"] = "none"
        elif override.lower() == "cuda":
            cfg["parser"]["device"] = 0           # GPU 0
            cfg["parser"]["device_map"] = "auto"

    return cfg