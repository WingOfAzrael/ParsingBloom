from dataclasses import dataclass
from datetime import datetime

@dataclass
class Transaction:
    transaction_id: int = None
    timestamp: datetime = None
    account_name: str = ""
    institution: str = ""
    account_number: str = ""
    external_entity: str = ""
    amount: float = 0.0
    available_balance: float = 0.0
    currency: str = ""
    description: str = ""
    transaction_type: str = ""    # personal / business / unclassified
    source_email: str = ""
    email_id: str = ""
    run_id: str = ""             # LINK to scraping run

    def to_dict(self):
        return {
            "transaction_id": self.transaction_id,
            "date": self.timestamp.strftime("%Y-%m-%d"),
            "internal_account_number": self.account_number,
            "internal_entity": self.account_name,
            "institution": self.institution,
            "external_entity": self.external_entity,
            "amount": self.amount,
            "available_balance": self.available_balance,  # Placeholder for available balance
            "currency": self.currency,
            "description": self.description,
            "transaction_type": self.transaction_type,
            "source_email": self.source_email,
            "email_id": self.email_id,
            "run_id": self.run_id
        }