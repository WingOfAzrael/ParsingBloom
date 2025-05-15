#utils.schema_seed (Pydantic-v2)
#----------------------------------
#Dynamically builds a Pydantic model from YAML schema definitions stored in src/schemas/.
#This allows for easy updates to the schema without changing the code.

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, create_model, field_validator
import dateutil.parser as du

# Root of the repo → src/ → utils/ (this file) → ../schemas
_SCHEMAS_DIR = Path(__file__).with_suffix("").parent.parent / "schemas"

_TYPE_MAP = {
    "str": str,
    "number": float,
    "date": date,
}


def _load_yaml(schema_name: str) -> dict:
    path = _SCHEMAS_DIR / f"{schema_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {path}")
    return yaml.safe_load(path.read_text())


def build_model(schema_name: str) -> type[BaseModel]:
    """Return a Pydantic-v2 model class for the given YAML schema."""
    schema = _load_yaml(schema_name)
    fields_cfg = schema["fields"]
    model_fields = {}

    for fname, fcfg in fields_cfg.items():
        py_type = _TYPE_MAP[fcfg["type"]]
        required = fcfg.get("required", False)
        default = ... if required else None
        model_fields[fname] = (py_type | None, default)

    Model = create_model(
        f"{schema_name.capitalize()}Record",
        **model_fields,
        __base__=BaseModel,
    )

    # ------------------------------------------------------------------ #
    # Generic coercion rules with Pydantic-v2 field_validator             #
    # ------------------------------------------------------------------ #
    @field_validator("*", mode="before")
    def _coerce(cls, v, info):
        if v is None:
            return v
        expected = info.annotation
        # strip union None: <class 'float | NoneType'> → float
        if getattr(expected, "__origin__", None) is not None:
            expected = expected.__args__[0]

        if expected is float and isinstance(v, str):
            return float(v.replace(",", "").replace(" ", ""))
        if expected is date and isinstance(v, str):
            return du.parse(v).date()
        return v

    return Model