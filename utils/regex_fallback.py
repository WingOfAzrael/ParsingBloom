# utils/regex_fallback.py

import re
from datetime import datetime
from config.config_loader import load_config

CFG = load_config("config/config.yaml")
SYN = CFG["parser"].get("synonyms", {})

# compile generic patterns
_DATE_RE     = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")
_AMT_RE      = re.compile(r"(?P<amount>[+-]?(?:R|ZAR)?\s?[\d,]+\.\d{2})")
_LAST4_RE    = re.compile(r"\*{2,}(\d{4})")
_BAL_RE      = re.compile(r"Available\s*balance[^\d]*(?:R|ZAR)?\s*(?P<bal>[\d,]+\.\d{2})", re.I)

# build “label: value” patterns from your synonyms, if any
def _label_pattern(key: str):
    alts = SYN.get(key, [])
    if not alts:
        return None
    esc = [re.escape(a) for a in alts]
    return re.compile(rf"(?P<label>{'|'.join(esc)})\s*:\s*(?P<val>.+)", re.I)

_INT_LBL = _label_pattern("internal_entity")
_EXT_LBL = _label_pattern("external_entity")


def dispatch(text: str) -> dict:
    """
    A bank-agnostic, generic regex fallback.  Returns a partial
    dict of any fields it could pull out.  Your LLM parser will
    fill in the rest or fall back to this.
    """
    data = {}

    # 1) explicit “label: value” lines from synonyms
    if _INT_LBL:
        for m in _INT_LBL.finditer(text):
            data["account_name"] = m.group("val").strip()
    if _EXT_LBL:
        for m in _EXT_LBL.finditer(text):
            data["external_entity"] = m.group("val").strip()

    # 2) ISO date (first match)
    m = _DATE_RE.search(text)
    if m:
        data["date"] = m.group("date")

    # 3) first currency amount → amount
    m = _AMT_RE.search(text)
    if m:
        amt = m.group("amount").replace("R", "").replace("ZAR", "").replace(",", "")
        try:
            data["amount"] = float(amt)
        except ValueError:
            pass

    # 4) masked account last-4
    m = _LAST4_RE.search(text)
    if m:
        data["account_number"] = m.group(1)

    # 5) available balance
    m = _BAL_RE.search(text)
    if m:
        bal = m.group("bal").replace(",", "")
        try:
            data["available_balance"] = float(bal)
        except ValueError:
            pass

    # 6) if we still have no external_entity, grab first non-blank line
    if "external_entity" not in data:
        for line in text.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # skip if it looks like a date or amount line
            if _DATE_RE.search(line) or _AMT_RE.search(line):
                continue
            data["external_entity"] = line[:50]
            break

    return data