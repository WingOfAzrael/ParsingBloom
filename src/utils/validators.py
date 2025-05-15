#import jsonschema
#from jsonschema import ValidationError
#
#TXN_SCHEMA = {
#    "$schema": "http://json-schema.org/draft-07/schema#",
#    "type": "object",
#    "required": [
#        "date",
#        "internal_account_number",   
#        "internal_entity",           
#        "institution",
#        "external_entity",
#        "amount",
#        "currency",                  
#        "description"
#    ],
#    "properties": {
#        "date":   {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
#        "internal_account_number": {"type": "string"},
#        "internal_entity":         {"type": "string"},
#        "institution":             {"type": "string"},
#        "external_entity":         {"type": ["string", "null"]},
#        "amount":                  {"type": "number"},
#        "available_balance":       {"type": ["number", "null"]},
#        "currency":                {"type": "string"},
#        "description":             {"type": "string"},
#        # columns you add further down the pipeline:
#        "transaction_type":        {"type": ["string", "null"]},
#        "source_email":            {"type": ["string", "null"]},
#        "message_id":                {"type": ["string", "null"]},
#        "run_id":                  {"type": ["string", "null"]}
#    },
#    "additionalProperties": False
#}
#
# Validates the given transaction dict against our JSON schema. Raises jsonschema.ValidationError if anything is missing or wrong.
#def validate_record(data: dict) -> None:
#
#    
#    try:
#        jsonschema.validate(instance=data, schema=TXN_SCHEMA)
#    except ValidationError as e:
#        # Re-raise with a clearer message
#        raise ValidationError(
#            f"Transaction data validation error: {e.message}"
#        )
    

#Exports two helpers:

# 1 - get_model(schema_name)  → dynamic Pydantic model class
# 2 - validate_record(data, schema="transactions") → dict (raises ValidationError)


from functools import lru_cache
from pydantic import ValidationError
from .schema_seed import build_model


@lru_cache(maxsize=None)
def get_model(schema_name: str):
    return build_model(schema_name)


def validate_record(data: dict, schema_name: str = "transactions") -> dict:
    Model = get_model(schema_name)
    return Model.model_validate(data).model_dump()