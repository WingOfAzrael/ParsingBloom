# utils/regex_fallback.py

import re
from datetime import datetime
from typing import Optional, Dict, Any
import logging
from core.config_loader import load_config
from parser.email_utils import extract_body

CFG = load_config()
SYN = CFG.synonyms    # ← now uses attribute, not indexing

logger = logging.getLogger(__name__)
# compile generic patterns

# ─── currency / amount helpers ────────────────────────────────────────────────
_CCY = r"[A-Z]{3}|[\u0024\u00A3\u20A0-\u20CF\u20B5\u20BD\u20BF]"        # no capture
_SEP = r"[ \u00A0\u2000-\u2060]?"                                       # space / thin-space / NBSP
_NUM = r"\d{1,3}(?:[ ,]\d{3})*(?:[.,]\d{2})|\d+(?:[.,]\d{2})"

_CCY_AMT_RE = re.compile(
    rf"(?P<ccy_left>{_CCY}){_SEP}(?P<amt_left>{_NUM})"                  # USD 1 234.56
    rf"|(?P<amt_right>{_NUM}){_SEP}(?P<ccy_right>{_CCY})",              # 1 234.56 USD
    re.I,
)

_DATE_RE  = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")
_AMT_RE   = _CCY_AMT_RE
_LAST4_RE = re.compile(r"\*{2,}(\d{4})")

_BAL_RE = re.compile(
    rf"(?:Available\s*balance|Balance\s*after\s*.*?transaction|New\s*balance)"
    rf"[^\d]*(?P<bal_ccy>{_CCY})?{_SEP}(?P<bal>{_NUM})",
    re.I,
)

# build “label: value” patterns from your synonyms, if any
def _label_pattern(key: str):
    alts = SYN.get(key, [])
    if not alts:
        return None
    esc = [re.escape(a) for a in alts]
    return re.compile(rf"(?P<label>{'|'.join(esc)})\s*:\s*(?P<val>.+)", re.I)

_INT_LBL = _label_pattern("internal_entity")
_EXT_LBL = _label_pattern("external_entity")


def dispatch(raw_email: Any, pdf_text: Optional[str] = None) -> Dict[str, Any]:
    """
    A bank-agnostic, generic regex fallback.
    Accepts:
      - raw_email: either the full Gmail API message dict or a plain text string
      - pdf_text:  optional extracted PDF text to append
    Returns a partial dict of any fields it could pull out.
    """
    # 1) determine the text we’ll run our regexes against
    if isinstance(raw_email, dict):
        # pull out the email body if you passed us the full message
        text = extract_body(raw_email)
    else:
        # assume it’s already the plain‐text body
        text = raw_email or ""

    # if you also passed PDF‐extracted text, tack it on
    if pdf_text:
        text = f"{text}\n\n{pdf_text}"

    data: Dict[str, Any] = {}

    # 2) explicit “label: value” lines via your synonyms
    if _INT_LBL:
        for m in _INT_LBL.finditer(text):
            data["account_name"] = m.group("val").strip()
    if _EXT_LBL:
        for m in _EXT_LBL.finditer(text):
            data["external_entity"] = m.group("val").strip()

    # 3) ISO date (first match)
    m = _DATE_RE.search(text)
    if m:
        data["date"] = m.group("date")

    # 4) first currency amount → amount (+ currency, if present)
    m = _AMT_RE.search(text)
    if m:
        # -------- amount ------------------------------------------------------
        amt_raw = m.group("amt_left") or m.group("amt_right")
        if amt_raw:
            amt_raw = re.sub(r"[ \u00A0\u2000-\u2060]", "", amt_raw)
            amt_raw = amt_raw.replace(",", "") if "." in amt_raw else amt_raw.replace(",", ".")
            try:
                data["amount"] = float(amt_raw)
            except ValueError:
                pass

        # -------- currency ----------------------------------------------------
        ccy = (m.group("ccy_left") or m.group("ccy_right") or "").strip()
        if ccy:
            data["currency"] = ccy.upper()

    # 5) masked account last-4
    m = _LAST4_RE.search(text)
    if m:
        data["account_number"] = m.group(1)

    # 6) available balance (+ default currency if we still don't have one)
    m = _BAL_RE.search(text)
    if m:
        bal_raw = re.sub(r"[ \u00A0\u2000-\u2060]", "", m.group("bal"))
        bal_raw = bal_raw.replace(",", "") if "." in bal_raw else bal_raw.replace(",", ".")
        try:
            data["available_balance"] = float(bal_raw)
        except ValueError:
            pass

        if "currency" not in data:
            ccy = (m.group("bal_ccy") or "").strip()
            if ccy:
                data["currency"] = ccy.upper()

    # 7) if we still have no external_entity, grab first non-blank line
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