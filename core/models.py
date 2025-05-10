# File: core/models.py

from decimal import Decimal
from datetime import date
from typing import Optional
from pydantic import BaseModel, constr


class Transaction(BaseModel):
    transaction_id: Optional[int] = None
    date: date
    internal_account_number: constr(min_length=1)
    internal_entity: constr(min_length=1)
    institution: constr(min_length=1)
    external_entity: constr(min_length=1)
    amount: Decimal
    available_balance: Optional[Decimal]
    currency: constr(regex=r"^[A-Z]{3}$")
    description: constr(min_length=1)
    transaction_type: constr(min_length=1)
    source_email: constr(min_length=1)
    email_id: constr(min_length=1)
    run_id: constr(min_length=1)

    def to_dict(self) -> dict:
        """
        Export exactly in your CSV schema format.
        """
        return {
            "date": self.date.isoformat(),
            "internal_account_number": self.internal_account_number,
            "internal_entity": self.internal_entity,
            "institution": self.institution,
            "external_entity": self.external_entity,
            "amount": str(self.amount),
            "available_balance": str(self.available_balance or ""),
            "currency": self.currency,
            "description": self.description,
            "transaction_type": self.transaction_type,
            "source_email": self.source_email,
            "email_id": self.email_id,
            "run_id": self.run_id,
        }