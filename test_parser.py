import base64
from parser.llm_parser import LLMParser
from gmail.client import GmailClient

# 1) Your raw email text, verbatim
sample_body = """Card payment

APPLE.COM/BILL ITUNES.CO – R 79.99

From ***7844

Card ending ***5441

Wednesday, 23 April at 10:41

Available balance: R 112.94

For more info, call 0800 07 96 97
"""

# 2) Base64-encode it the way Gmail API would
encoded = base64.urlsafe_b64encode(sample_body.encode()).decode()

# 3) Build the fake Gmail message
msg = {
    "id": "test-1",
    "payload": {
        "parts": [
            {"mimeType": "text/plain", "body": {"data": encoded}}
        ],
        "headers": [
            {"name": "From", "value": "notifications@standardbank.com"}
        ]
    }
}

parser = LLMParser()
txns = parser.parse(msg)

print("At least we get here:")
print(txns)