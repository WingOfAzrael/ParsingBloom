# ===== connectors/base.py =====
from typing import Protocol, Any, List, Dict

class Connector(Protocol):
    def authenticate(self) -> None:
        """Perform any auth (OAuth, tokens, API keys)."""

    def fetch_messages(
        self,
        *,
        since: str,
        max_results: int,
        query: str
    ) -> List[Dict[str, Any]]:
        """Return a list of raw message dicts."""

    def fetch_pdf_attachment(
        self,
        msg: Dict[str, Any],
        attachment_id: str
    ) -> bytes | None:
        """Download raw bytes of a PDF attachment, or None."""