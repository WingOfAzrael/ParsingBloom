import base64
import html
import re
from typing import Dict
from bs4 import BeautifulSoup


def extract_body(msg: Dict) -> str:
    """
    Walk the full multipart tree, find the first text/plain or text/html part,
    decode it, strip HTML if needed, collapse whitespace.
    If nothing found, return msg['snippet'].
    """
    def dfs(part):
        mt = part.get("mimeType", "")
        # plain‐text wins
        if mt == "text/plain":
            return part.get("body", {}).get("data", ""), "plain"
        # next best is HTML
        if mt == "text/html":
            return part.get("body", {}).get("data", ""), "html"
        # else recurse into subparts
        for sub in part.get("parts", []):
            data, kind = dfs(sub)
            if data:
                return data, kind
        return None, None

    payload = msg.get("payload", {})
    data_b64, kind = dfs(payload)

    if data_b64:
        decoded = base64.urlsafe_b64decode(data_b64.encode("utf-8")).decode("utf-8", errors="ignore")
        if kind == "html":
            text = BeautifulSoup(decoded, "html.parser").get_text(" ", strip=True)
        else:
            text = decoded
        # unescape entities & collapse whitespace
        text = html.unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    # fallback to Gmail snippet (which you said *does* have the info)
    return msg.get("snippet", "")

def _decode(data_b64: str) -> str:
    """
    Decode Gmail URL-safe base64 to utf-8 string.
    """
    decoded_bytes = base64.urlsafe_b64decode(data_b64.encode("utf-8"))
    return decoded_bytes.decode("utf-8", errors="ignore")

def _strip_html(html_src: str) -> str:
    """
    Convert HTML to plain text via BeautifulSoup for robust extraction:
    - Preserves spacing between tags
    - Unescapes HTML entities
    - Collapses excess whitespace
    """
    soup = BeautifulSoup(html_src, "html.parser")
    text = soup.get_text(" ", strip=True)      # join blocks with spaces
    text = html.unescape(text)                 # turn &amp; → &
    return re.sub(r"\s+", " ", text).strip()   # collapse runs of whitespace