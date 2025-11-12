import json

import json_repair
from jsonschema import ValidationError, validate


class ModelJSONDecodeError(json.JSONDecodeError):
    """Custom JSONDecodeError for model response parsing errors."""


class ModelJSONSchemaValidationError(Exception):
    """Custom exception for JSON schema validation errors."""

    def __init__(self, message: str, current_structure=None, expected_schema=None):
        super().__init__(message)
        self.current_structure = current_structure
        self.expected_schema = expected_schema

    def __str__(self):
        msg = super().__str__()
        if self.current_structure is not None:
            msg += (
                f"\nCurrent structure: {json.dumps(self.current_structure, indent=2)}"
            )
        if self.expected_schema is not None:
            msg += f"\nExpected schema: {json.dumps(self.expected_schema, indent=2)}"
        return msg


# A safe JSON loader for model responses that attempts to repair malformed JSON strings
def safe_model_json_loads(payload: str):
    if not payload:
        raise ModelJSONDecodeError("Empty JSON payload", payload, 0)

    repaired_object = json_repair.loads(payload)
    if repaired_object:
        return repaired_object
    else:
        raise ModelJSONDecodeError("Empty JSON after repair", payload, 0)


def json_schema_validation(payload: str, schema: dict):
    """Validate JSON payload against a given schema."""
    try:
        data = safe_model_json_loads(payload)
        validate(instance=data, schema=schema)
        return data
    except ValidationError as e:
        raise ModelJSONSchemaValidationError(
            f"JSON schema validation error: {str(e)}",
            current_structure=data,
            expected_schema=schema,
        ) from e
    except ModelJSONDecodeError:
        raise
