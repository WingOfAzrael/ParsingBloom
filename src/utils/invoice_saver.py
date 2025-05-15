# utils/invoice_saver.py

from pathlib import Path
import base64
from typing import Dict, List

import keyring
from PyPDF2 import PdfReader, PdfWriter

from core.config_loader import load_config
from core.models import Transaction


# Load your PDF passwords from config/config.yaml under "pdf_passwords"
CFG = load_config()
PDF_PASSWORDS: Dict[str, str] = CFG.get("pdf_passwords", {})

BASE_DIR = Path("Invoices")


def save_attachments(
    gmail_service,
    message: Dict,
    txn: Transaction
) -> List[str]:
    """
    Downloads any PDF attachments, saves the encrypted version, then
    immediately writes out a decrypted copy using the configured password.
    Returns a list of saved file paths (decrypted if a password was found,
    otherwise the original encrypted file).
    """
    saved: List[str] = []

    # Walk all MIME parts
    def _walk_parts(part: Dict) -> List[Dict]:
        parts = [part]
        for sub in part.get("parts", []):
            parts.extend(_walk_parts(sub))
        return parts

    all_parts = _walk_parts(message.get("payload", {}))

    for part in all_parts:
        fname = part.get("filename", "")
        aid   = part.get("body", {}).get("attachmentId")
        if not fname or not aid or not fname.lower().endswith(".pdf"):
            continue

        # 1) Download encrypted bytes
        att = (
            gmail_service.users()
            .messages()
            .attachments()
            .get(userId="me", messageId=message["id"], id=aid)
            .execute()
        )
        data = base64.urlsafe_b64decode(att.get("data", ""))

        # 2) Determine where to save
        bucket = "Company" if txn.transaction_type == "business" else "Personal"
        year, month = str(txn.timestamp.year), f"{txn.timestamp.month:02d}"
        folder = BASE_DIR / bucket / year / month

        # ←── MAKE SURE THE DIR EXISTS ───────────────────────────────
        folder.mkdir(parents=True, exist_ok=True)

        enc_path = folder / fname
        with open(enc_path, "wb") as f:
            f.write(data)

        # 3) Look up password
        pwd = PDF_PASSWORDS.get(txn.institution)
        if not pwd:
            pwd = keyring.get_password("email_agent", txn.institution)

        # 4) Decrypt if we have a password
        if pwd:
            dec_path = enc_path.with_name(f"decrypted_{fname}")
            try:
                _decrypt_pdf(enc_path, dec_path, pwd)
                saved.append(str(dec_path))
            except Exception:
                # decryption failed—keep encrypted
                saved.append(str(enc_path))
        else:
            saved.append(str(enc_path))

    return saved


def _decrypt_pdf(src: Path, dest: Path, password: str) -> None:
    """
    Opens the encrypted PDF at `src` with `password` and writes
    an un-encrypted copy to `dest`. Raises on failure.
    """
    reader = PdfReader(str(src))
    if reader.is_encrypted:
        reader.decrypt(password)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    # ensure parent exists again (should already from above)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        writer.write(f)