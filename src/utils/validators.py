
from functools import lru_cache
from pydantic import ValidationError
from .schema_seed import build_model
import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, model_validator, ValidationError
from core.config_loader import load_config

# ─── 1) Load config & build helpers ─────────────────────────────────────────
cfg        = load_config()
mask_re    = re.compile(cfg.account_mask.regex)
suffix_len = cfg.account_mask.suffix_length
accounts   = cfg.accounts

# build suffix → account map
_account_map = {
    str(acc.internal_account_number)[-suffix_len:]: acc
    for acc in accounts
}

# cache allowed names/institutions
_ALLOWED_ENTITIES    = {acc.internal_entity for acc in accounts}
_ALLOWED_INSTITUTIONS = {acc.institution for acc in accounts}

# ─── get the dynamic model with *all* YAML fields ──────────────────
BaseTxModel = build_model("transactions")

class Transaction(BaseTxModel):
    # ignore any extra fields
    model_config = ConfigDict(extra="ignore")

    # still annotate as str so Pylance is happy
    internal_account_number: str
    internal_entity:         str
    institution:             str

    #external_account_number: Optional[str] = None
    external_entity:         Optional[str] = None

    transaction_type: str
    description:      str
    run_id: str

    @model_validator(mode="before")
    def _resolve_accounts(cls, values: dict) -> dict:
        # A) normalize numeric values or numeric‐looking strings → pure digit‐strings
        for side in ("internal", "external"):
            key = f"{side}_account_number"
            v = values.get(key)
            if v is None:
                continue
            # a) real numbers → int → str
            if isinstance(v, (int, float)):
                values[key] = str(int(v))
                continue
            # b) strings like "12345.0" → strip ".0"
            if isinstance(v, str) and v.endswith(".0") and v[:-2].isdigit():
                values[key] = v[:-2]
                continue
            # otherwise leave it as-is (e.g. merchant account suffixes, fully redacted masks, etc.)
            values[key] = v

        tx_type = values.get("transaction_type", "").lower()

        # C) If it's a TRANSFER, do BOTH-SIDED suffix-mapping and exit
        if tx_type == "transfer":
            desc  = values.get("description", "") or ""
            masks = mask_re.findall(desc)
            if len(masks) >= 2:
                src_suf = masks[0]; dst_suf = masks[1]
                our_suf = values.get("internal_account_number", "")[-suffix_len:]
                other   = dst_suf if src_suf == our_suf else src_suf
                src_acc = _account_map.get(our_suf)
                dst_acc = _account_map.get(other)
                if src_acc and dst_acc:
                    values["internal_account_number"] = str(src_acc.internal_account_number)
                    values["internal_entity"]         = src_acc.internal_entity
                    values["institution"]             = src_acc.institution
                    values["external_account_number"] = str(dst_acc.internal_account_number)
                    values["external_entity"]         = dst_acc.internal_entity
            return values
        # D) NON-TRANSFER: first map the INTERNAL side only
        acct = values.get("internal_account_number")
        if acct:
            for acc in accounts:
                full = str(acc.internal_account_number)
                if acct == full or full.endswith(acct):
                    values["internal_entity"] = acc.internal_entity
                    values["institution"]      = acc.institution
                    break
        # E) NON-TRANSFER fallback for merchant: fix external_entity if it echoes our bank
        ext  = values.get("external_entity", "") or ""
        inst = values.get("institution", "") or ""
        if inst and ext.startswith(inst):
            desc  = values.get("description", "") or ""
            parts = desc.split(" ", 2)
            # drop transaction type and possible verb, grab merchant name
            if len(parts) >= 3:
                values["external_entity"] = parts[2]
            elif len(parts) == 2:
                values["external_entity"] = parts[1]

        return values

    @model_validator(mode="after")
    def _validate_enums(cls, m: "Transaction") -> "Transaction":
        """
        Ensure that after resolution, internal_entity and institution
        are actually one of the configured values.
        """
        if m.internal_entity not in _ALLOWED_ENTITIES:
            raise ValidationError(
                f"internal_entity {m.internal_entity!r} not in config", cls
            )
        if m.institution not in _ALLOWED_INSTITUTIONS:
            raise ValidationError(
                f"institution {m.institution!r} not in config", cls
            )
        return m


@lru_cache(maxsize=None)
def get_model(schema_name: str):
    # For the “transactions” schema, use our Transaction class
    if schema_name.lower() in ("transactions", "transaction"):
        return Transaction
    # Otherwise fall back to the dynamic YAML-based model
    return build_model(schema_name)


#def validate_record(data: dict, schema_name: str = "transactions") -> dict:
def validate_record(data: dict, schema_name: str = "transactions") -> BaseModel:
    #Model = get_model(schema_name)
    #return Model.model_validate(data).model_dump()

    """
    Parse and validate into a Pydantic model instance.
    """
    Model = get_model(schema_name)
    return Model.model_validate(data)