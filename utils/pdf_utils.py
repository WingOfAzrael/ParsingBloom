# ===== utils/pdf_utils.py =====

import base64
import io
import logging
from typing import List, Optional

import pdfplumber
from pdfplumber.pdf import PdfminerException
from googleapiclient.discovery import Resource

from config.config_loader import load_config

# Load configuration once
CFG = load_config("config/config.yaml")


def extract_pdf_texts(service: Resource, message: dict) -> Optional[str]:
    """
    Fetch PDF attachments from a Gmail message, open them with the configured password (if any),
    and return all their text concatenated. Returns None on any open or decode error.
    """
    pwd = CFG.get("scraper", {}).get("pdf_password") or None
    logging.debug("PDF extractor using password: %s", "***" if pwd else "<no password>")

    texts: List[str] = []
    for part in message.get("payload", {}).get("parts", []):
        fname = part.get("filename", "").lower()
        aid   = part.get("body", {}).get("attachmentId")
        if not fname.endswith(".pdf") or not aid:
            continue

        # download the raw bytes from Gmail
        try:
            data = (
                service.users()
                       .messages()
                       .attachments()
                       .get(userId="me", messageId=message["id"], id=aid)
                       .execute()
                       .get("data", "")
            )
            raw = base64.urlsafe_b64decode(data)
        except Exception as e:
            logging.warning("Failed to download PDF attachment: %s", e)
            return None

        # only pass password if it's non-empty
        open_kwargs = {}
        if pwd:
            open_kwargs["password"] = pwd

        try:
            with pdfplumber.open(io.BytesIO(raw), **open_kwargs) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        texts.append(text)
        except (PdfminerException, Exception) as e:
            logging.warning("Could not open PDF (wrong password or corrupt): %s", e)
            return None

    return "\n\n".join(texts) if texts else None


def extract_text_from_bytes(raw: bytes) -> Optional[str]:
    """
    Given raw PDF bytes, open them with the configured password (if any)
    and return all extracted text concatenated. Returns None on failure.
    """
    pwd = CFG.get("scraper", {}).get("pdf_password") or None
    logging.debug("Byte‐based PDF parser using password: %s", "***" if pwd else "<no password>")

    open_kwargs = {}
    if pwd:
        open_kwargs["password"] = pwd

    texts: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(raw), **open_kwargs) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
    except (PdfminerException, Exception) as e:
        logging.warning("Could not open PDF bytes: %s", e)
        return None

    return "\n\n".join(texts) if texts else None