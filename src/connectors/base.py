# File: connectors/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Connector(ABC):
    """
    Base class for all connectors (Gmail, trading API, etc).
    Subclasses must accept a single `cfg: dict` in __init__.
    """

    @abstractmethod
    def authenticate(self) -> None:
        """
        Perform any authentication or token refresh logic.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_messages(
        self,
        *,
        since: str,
        max_results: int,
        query: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch raw message dicts from the external service.
        """
        raise NotImplementedError

    @abstractmethod
    def fetch_pdf_attachment(
        self,
        msg: Dict[str, Any],
        attachment_id: str
    ) -> Optional[bytes]:
        """
        If the message has a PDF attachment, return its bytes.
        """
        raise NotImplementedError