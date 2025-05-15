# File: parser/base.py

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.models import Transaction


class Parser(ABC):
    """
    Abstract base class for parsers.
    Each parser must implement parse(), which takes:
      - raw_email: the Gmail API message dict
      - pdf_text: optional extracted PDF text
    and returns a validated Transaction.
    """

    @abstractmethod
    def parse(self, raw_email: Dict[str, Any], pdf_text: Optional[str] = None) -> Transaction:
        """
        Parse the given raw email dict (and optional PDF text)
        into a Transaction object.
        """
        raise NotImplementedError("Parser subclasses must implement parse()")