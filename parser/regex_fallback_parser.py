# parser/regex_fallback_parser.py

import logging
from typing import Optional, Dict, Any
from parser.base import Parser
from parser.email_utils import extract_body
from utils.regex_fallback import dispatch as regex_dispatch
from core.models import Transaction

class RegexFallbackParser(Parser):
    """
    Fallback parser that only uses your regex-based extractor.
    """

    def parse(self, raw_email: Dict[str, Any], pdf_text: Optional[str] = None) -> Transaction:
        body      = extract_body(raw_email)
        full_text = body + ("\n\n" + pdf_text if pdf_text else "")

        logging.getLogger(__name__).debug(
            "RegexFallbackParser input",
            extra={"email_id": raw_email.get("id"), "text": full_text[:200]}
        )

        data = regex_dispatch(full_text)
        logging.getLogger(__name__).debug(
            "RegexFallbackParser output",
            extra={"email_id": raw_email.get("id"), "data": data}
        )

        txn = Transaction.parse_obj(data)
        logging.getLogger(__name__).info(
            "RegexFallbackParser success",
            extra={"email_id": raw_email.get("id"), "parsed": txn.dict()}
        )
        return txn
