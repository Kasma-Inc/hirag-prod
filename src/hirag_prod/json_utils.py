import json
from typing import Any

import json_repair
from jsonschema import ValidationError, validate
from jsonschema.validators import validator_for


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
def safe_model_json_loads(payload: str) -> Any:
    if not payload:
        raise ModelJSONDecodeError("Empty JSON payload", payload, 0)

    repaired_object = json_repair.loads(payload)
    if repaired_object:
        return repaired_object
    else:
        raise ModelJSONDecodeError("Empty JSON after repair", payload, 0)


def json_schema_validation(payload: str, schema: dict) -> Any:
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


# Convenience helpers for OpenAI-style strict JSON generation
def build_openai_json_response_format(
    schema: dict,
    *,
    name: str = "response",
    strict: bool = True,
) -> dict:
    """Build OpenAI chat.completions response_format for a JSON Schema.

    Returns a dict like:
    {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": strict, "schema": schema}
    }

    for schema pls refer to: https://json-schema.org/understanding-json-schema/about
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": strict,
            "schema": schema,
        },
    }


def openai_json_response_format_validation(response_format: dict) -> bool:
    """Validate an OpenAI chat.completions JSON response_format wrapper.

    Expects a dict of the form:
    {
        "type": "json_schema",
        "json_schema": {"name": str, "strict": bool, "schema": dict}
    }

    Returns True if both the wrapper structure and the inner JSON Schema are valid.
    """
    try:
        if not isinstance(response_format, dict):
            return False
        if response_format.get("type") != "json_schema":
            return False
        js = response_format.get("json_schema")
        if not isinstance(js, dict):
            return False
        name = js.get("name")
        strict = js.get("strict")
        schema = js.get("schema")
        if (
            not isinstance(name, str)
            or not isinstance(strict, bool)
            or not isinstance(schema, dict)
        ):
            return False

        # Validate the inner JSON Schema itself using jsonschema
        validator_cls = validator_for(schema)
        validator_cls.check_schema(schema)

        return True
    except Exception:
        return False
