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

from typing import List, Dict, Any, Optional
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
        Fetch up to `max_results` messages matching `query` since `since`.
        Paginates through Gmail's list API (max 500 IDs per call).
        """
        logger = logging.getLogger(__name__)
        logger.info(
            "Starting fetch_messages",
            extra={"since": since, "max_results": max_results, "query": query}
        )

        # Global email-filter settings
        cfg = load_config()
        ef  = cfg.email_filter

        # Build Gmail q-string
        q_parts: List[str] = []
        if ef.use_label and ef.label_name:
            q_parts.append(f"label:{ef.label_name}")
        if ef.use_domain_filter and ef.sender_csv:
            try:
                with open(ef.sender_csv, newline='') as f:
                    domains = [row[0].strip() for row in csv.reader(f) if row]
                if domains:
                    domain_q = " OR ".join(f"from:{d}" for d in domains)
                    q_parts.append(f"({domain_q})")
            except Exception as e:
                logger.warning(
                    "Failed loading sender_csv for domain filter",
                    extra={"error": str(e), "file": str(ef.sender_csv)}
                )
        if query:
            q_parts.append(query)
        if since:
            q_parts.append(f"after:{since}")

        params: Dict[str, Any] = {
            "userId": "me",
            # Gmail list() honors up to 500; we'll page if max_results > 500
            "maxResults": min(max_results, 500)
        }
        if q_parts:
            params["q"] = " ".join(q_parts)

        all_msgs: List[Dict[str, Any]] = []
        next_token: Optional[str] = None

        while True:
            if next_token:
                params["pageToken"] = next_token

            resp = self.service.users().messages().list(**params).execute()
            ids = resp.get("messages", [])
            logger.info(
                "Listed message IDs",
                extra={"count": len(ids), "query": params.get("q")}
            )

            # Fetch each message
            for m in ids:
                if len(all_msgs) >= max_results:
                    break
                msg = (
                    self.service.users()
                                .messages()
                                .get(userId="me", id=m["id"], format="full")
                                .execute()
                )
                all_msgs.append(msg)
                logger.debug(
                    "Fetched single message",
                    extra={"index": len(all_msgs), "id": m["id"]}
                )

            # Stop if we reached the user’s limit
            if len(all_msgs) >= max_results:
                break

            # Otherwise, continue if there’s another page
            next_token = resp.get("nextPageToken")
            if not next_token:
                break

        return all_msgs

    def fetch_pdf_attachment(
        self,
        msg: Dict[str, Any],
        attachment_id: str
    ) -> Optional[bytes]:
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
