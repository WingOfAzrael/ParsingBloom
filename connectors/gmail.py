# ===== connectors/gmail.py =====
import os
import pickle
import base64
from typing import List, Dict, Any

import keyring
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from connectors.base import Connector

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailConnector(Connector):
    def __init__(self, cfg: dict):
        # cfg == CFG["connectors"]["gmail"]
        self.creds_path = cfg["credentials_path"]
        self.token_path = cfg["token_path"]
        self.service    = self._authenticate()

    def _authenticate(self):
        creds = None
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as f:
                creds = pickle.load(f)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.creds_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'wb') as f:
                pickle.dump(creds, f)
            keyring.set_password(
                "email_agent",
                "gmail_refresh_token",
                creds.refresh_token
            )

        return build('gmail', 'v1', credentials=creds)

    def authenticate(self) -> None:
        # no-op (auth happened in __init__)
        pass

    def fetch_messages(
        self,
        *,
        since: str,
        max_results: int,
        query: str
    ) -> List[Dict[str, Any]]:
        # Build Gmail q-string using the same logic as before
        q_parts = []
        if query:
            q_parts.append(query)
        if since:
            # Gmail expects YYYY/MM/DD
            q_parts.append(f"after:{since}")
        params = {"userId":"me", "maxResults": max_results}
        if q_parts:
            params["q"] = " ".join(q_parts)

        resp = self.service.users().messages().list(**params).execute()
        ids  = resp.get("messages", [])
        full = []
        for m in ids:
            full.append(
                self.service.users()
                            .messages()
                            .get(userId="me", id=m["id"], format="full")
                            .execute()
            )
        return full

    def fetch_pdf_attachment(
        self,
        msg: Dict[str, Any],
        attachment_id: str
    ) -> bytes | None:
        data = (
            self.service.users()
                        .messages()
                        .attachments()
                        .get(
                            userId="me",
                            messageId=msg["id"],
                            id=attachment_id
                        )
                        .execute()
                        .get("data", "")
        )
        return base64.urlsafe_b64decode(data) if data else None