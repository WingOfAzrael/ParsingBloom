import os
import base64

class InvoiceSaver:
    def __init__(self, base_dir: str = "Invoices"):
        self.base_dir = base_dir

    def save_attachments(self, gmail_service, message: dict, txn):
        parts = message.get("payload", {}).get("parts", [])
        for p in parts:
            fname = p.get("filename", "")
            aid   = p.get("body", {}).get("attachmentId")
            if not fname or not aid:
                continue

            att = gmail_service.users()\
                               .messages()\
                               .attachments()\
                               .get(userId="me",
                                    messageId=message["id"],
                                    id=aid)\
                               .execute()
            data = base64.urlsafe_b64decode(att.get("data","").encode("UTF-8"))

            dt     = txn.timestamp
            bucket= "Company" if txn.transaction_type=="business" else "Personal"
            year   = str(dt.year)
            month  = f"{dt.month:02d}"
            path   = os.path.join(self.base_dir, bucket, year, month)
            os.makedirs(path, exist_ok=True)

            with open(os.path.join(path, fname), "wb") as f:
                f.write(data)