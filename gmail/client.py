import os
import pickle
import yaml
import keyring
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class GmailClient:
    def __init__(self, config_path="config/config.yaml"):
        cfg = yaml.safe_load(open(config_path))
        self.creds_path = cfg['gmail']['credentials_path']
        self.token_path = cfg['gmail']['token_path']
        self.service = self._authenticate()

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
                    self.creds_path,
                    ['https://www.googleapis.com/auth/gmail.readonly']
                )
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'wb') as f:
                pickle.dump(creds, f)
            keyring.set_password("email_agent", "gmail_refresh_token", creds.refresh_token)
        return build('gmail', 'v1', credentials=creds)

    def fetch_messages(self, query, after=None, max_results=100):
        # Build a proper q-string from non-empty components
        q_parts = []
        if query:
            q_parts.append(query)
        if after:
            q_parts.append(f"after:{after}")
        q = " ".join(q_parts)

        # Only include 'q' when it’s actually non-empty
        params = {
            "userId": "me",
            "maxResults": max_results,
        }
        if q:
            params["q"] = q

        resp = self.service.users().messages().list(**params).execute()
        msgs = resp.get('messages', [])
        full = []
        for m in msgs:
            full.append(
                self.service.users()
                            .messages()
                            .get(userId='me', id=m['id'], format='full')
                            .execute()
            )
        return full