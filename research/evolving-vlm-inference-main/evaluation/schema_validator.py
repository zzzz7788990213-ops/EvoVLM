"""Small fail-closed executor for the JSON Schema subset used by Gate 1 v3."""

from __future__ import annotations

import math
import re
from typing import Any


class SchemaError(AssertionError):
    pass


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise SchemaError(f"unsupported schema type: {expected}")


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise SchemaError(f"only local schema refs are supported: {reference}")
    current: Any = root
    for component in reference[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or component not in current:
            raise SchemaError(f"unresolved schema ref: {reference}")
        current = current[component]
    if not isinstance(current, dict):
        raise SchemaError(f"schema ref is not an object: {reference}")
    return current


def validate(instance: Any, schema: dict[str, Any]) -> None:
    _validate(instance, schema, schema, "$")


def _validate(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate(instance, _resolve_ref(root, schema["$ref"]), root, path)
        return
    if "anyOf" in schema:
        failures: list[str] = []
        for alternative in schema["anyOf"]:
            try:
                _validate(instance, alternative, root, path)
                break
            except SchemaError as exc:
                failures.append(str(exc))
        else:
            raise SchemaError(f"{path}: no anyOf branch matched: {failures}")
    if "const" in schema and instance != schema["const"]:
        raise SchemaError(f"{path}: expected const {schema['const']!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise SchemaError(f"{path}: value is outside enum")
    if "type" in schema:
        expected = schema["type"]
        types = [expected] if isinstance(expected, str) else expected
        if not isinstance(types, list) or not all(isinstance(item, str) for item in types):
            raise SchemaError(f"{path}: invalid type declaration")
        if not any(_type_matches(instance, item) for item in types):
            raise SchemaError(f"{path}: expected type {types}, got {type(instance).__name__}")
    if isinstance(instance, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in instance]
        if missing:
            raise SchemaError(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SchemaError(f"{path}: properties must be an object")
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], root, f"{path}.{key}")
            elif additional is False:
                raise SchemaError(f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                _validate(value, additional, root, f"{path}.{key}")
    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise SchemaError(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise SchemaError(f"{path}: more than maxItems")
        if schema.get("uniqueItems"):
            serialized = [repr(item) for item in instance]
            if len(serialized) != len(set(serialized)):
                raise SchemaError(f"{path}: array items are not unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, value in enumerate(instance):
                _validate(value, item_schema, root, f"{path}[{index}]")
    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise SchemaError(f"{path}: shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise SchemaError(f"{path}: longer than maxLength")
        if "pattern" in schema and re.fullmatch(schema["pattern"], instance) is None:
            raise SchemaError(f"{path}: string does not match pattern")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            raise SchemaError(f"{path}: non-finite number")
        if "minimum" in schema and instance < schema["minimum"]:
            raise SchemaError(f"{path}: below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise SchemaError(f"{path}: above maximum")
        if "exclusiveMinimum" in schema and instance <= schema["exclusiveMinimum"]:
            raise SchemaError(f"{path}: not above exclusiveMinimum")
        if "exclusiveMaximum" in schema and instance >= schema["exclusiveMaximum"]:
            raise SchemaError(f"{path}: not below exclusiveMaximum")
