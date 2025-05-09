import jsonschema
from jsonschema import ValidationError

TXN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": [
        "date",
        "account_name",
        "account_number",
        "institution",
        "external_entity",
        "amount",
        "currency",
        "description"
    ],
    "properties": {
        "date": { "type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$" },
        "account_name": { "type": "string" },
        "account_number": { "type": "string" },
        "institution": { "type": "string" },
        "external_entity": { "type": ["string", "null"] },
        "amount": { "type": "number" },
        "email_id": { "type": ["string", "null"] },   # ← allow blank / missing
        "available_balance": { "type": ["number", "null"] },
        "currency": { "type": "string" },
        "description": { "type": "string" }
    },
    "additionalProperties": False
}

# Validates the given transaction dict against our JSON schema. Raises jsonschema.ValidationError if anything is missing or wrong.
def validate_transaction_data(data: dict) -> None:

    
    try:
        jsonschema.validate(instance=data, schema=TXN_SCHEMA)
    except ValidationError as e:
        # Re-raise with a clearer message
        raise ValidationError(
            f"Transaction data validation error: {e.message}"
        )