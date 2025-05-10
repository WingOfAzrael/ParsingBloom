import os
import csv
import pickle
import base64
import logging
from typing import List, Dict, Any

import keyring
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

from connectors.base import Connector
from config.config_loader import ConnectorConfig, load_config

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailConnector(Connector):
    def __init__(self, cfg: ConnectorConfig):
        """
        cfg: a ConnectorConfig Pydantic object, with attributes:
          - credentials_path (Path)
          - token_path       (Path)
        """
        self.creds_path = cfg.credentials_path
        self.token_path = cfg.token_path
        self.service    = self._authenticate()
        logging.getLogger(__name__).info(
            "GmailConnector initialized",
            extra={"creds_path": str(self.creds_path), "token_path": str(self.token_path)}
        )

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
                    str(self.creds_path), SCOPES
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'wb') as f:
                pickle.dump(creds, f)
            # store refresh token in keyring
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
        """
        since: YYYY-MM-DD
        max_results & query must be provided by caller (e.g. CFG.scraper)
        """
        logging.getLogger(__name__).info(
            "Starting fetch_messages",
            extra={"since": since, "max_results": max_results, "query": query}
        )
        # Load global filter settings
        cfg = load_config()
        ef = cfg.email_filter

        q_parts: List[str] = []
        # Label filter
        if ef.use_label and ef.label_name:
            q_parts.append(f"label:{ef.label_name}")

        # Domain filter
        if ef.use_domain_filter and ef.sender_csv:
            try:
                with open(ef.sender_csv, newline='') as f:
                    reader = csv.reader(f)
                    domains = [row[0].strip() for row in reader if row]
                if domains:
                    domain_query = " OR ".join(f"from:{d}" for d in domains)
                    q_parts.append(f"({domain_query})")
            except Exception as e:
                logging.getLogger(__name__).warning(
                    "Failed loading sender_csv for domain filter",
                    extra={"error": str(e), "file": str(ef.sender_csv)}
                )

        # Standard query and date filters
        if query:
            q_parts.append(query)
        if since:
            q_parts.append(f"after:{since}")

        params: Dict[str, Any] = {"userId": "me", "maxResults": max_results}
        if q_parts:
            params["q"] = " ".join(q_parts)

        resp = self.service.users().messages().list(**params).execute()
        ids  = resp.get("messages", [])
        logging.getLogger(__name__).info(
            "Listed message IDs",
            extra={"count": len(ids), "query": params.get("q")} 
        )

        full_msgs: List[Dict[str,Any]] = []
        for idx, m in enumerate(ids, start=1):
            msg = (
                self.service.users()
                            .messages()
                            .get(userId="me", id=m["id"], format="full")
                            .execute()
            )
            full_msgs.append(msg)
            logging.getLogger(__name__).debug(
                "Fetched single message",
                extra={"index": idx, "id": m["id"]}
            )

        return full_msgs

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
        if data:
            return base64.urlsafe_b64decode(data)
        return None
