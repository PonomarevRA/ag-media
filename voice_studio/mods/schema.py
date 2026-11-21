"""Strict, dependency-free media mod v1 manifest and JSON value validation."""

from copy import deepcopy
import json
import math
from pathlib import PurePosixPath
import re

from .errors import ModError
from .contracts import resolve_contract

IDENTIFIER = re.compile(r"^[a-z][a-z0-9.-]{0,95}$")
VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:[-+][a-zA-Z0-9.-]+)?$")
SCHEMA_KEYS = {"type", "properties", "required", "additionalProperties", "enum", "items",
               "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems",
               "description", "title", "default", "enumLabels"}
TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


def identifier(value, label="identifier"):
    """Return a bounded lowercase namespace ID or raise ModError."""
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ModError(f"Invalid {label}")
    return value


def relative_path(value):
    """Validate a package-relative path without traversal or platform ambiguity."""
    if (not isinstance(value, str) or not value or len(value) > 240 or "\\" in value
            or ":" in value or "\0" in value or value.startswith("/")
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ModError("Invalid package-relative path")
    return PurePosixPath(value)


def entrypoint(value):
    """Validate one package-local Python file:function declaration without importing it."""
    if not isinstance(value, str) or value.count(":") != 1:
        raise ModError("Expected relative worker.py:function entrypoint")
    filename, function = value.split(":")
    relative_path(filename)
    if not filename.endswith(".py") or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", function):
        raise ModError("Invalid Python entrypoint")
    return value


def json_bytes(value):
    """Encode deterministic finite JSON; reject non-JSON types and NaN values."""
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as error:
        raise ModError(f"Expected finite JSON: {error}") from error


def validate_schema(schema, depth=0):
    """Reject unsupported schema keywords so declarations never promise ignored checks."""
    if not isinstance(schema, dict) or depth > 8 or set(schema) - SCHEMA_KEYS:
        raise ModError("Unsupported JSON schema (v1 supports a bounded typed subset)")
    if schema.get("type") not in TYPES:
        raise ModError("JSON schema requires a supported explicit type")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or len(properties) > 64:
        raise ModError("Schema properties must be a bounded object")
    for name, child in properties.items():
        if not isinstance(name, str) or not name or len(name) > 100:
            raise ModError("Invalid schema property name")
        validate_schema(child, depth + 1)
    required = schema.get("required", [])
    if (not isinstance(required, list) or any(not isinstance(key, str) for key in required)
            or not set(required) <= set(properties) or len(set(required)) != len(required)):
        raise ModError("Schema required keys must name declared unique properties")
    if type(schema.get("additionalProperties", False)) is not bool:
        raise ModError("additionalProperties must be a boolean")
    if "items" in schema:
        validate_schema(schema["items"], depth + 1)
    for key in ("minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems"):
        if key in schema:
            value = schema[key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ModError(f"Schema {key} must be finite")
            if key not in {"minimum", "maximum"} and (type(value) is not int or value < 0):
                raise ModError(f"Schema {key} must be a nonnegative integer")
    if "enum" in schema and (not isinstance(schema["enum"], list) or not schema["enum"]):
        raise ModError("Schema enum must be a nonempty list")
    if "enumLabels" in schema:
        labels = schema["enumLabels"]
        if ("enum" not in schema or not isinstance(labels, list) or len(labels) != len(schema["enum"])
                or any(not isinstance(label, str) or not 1 <= len(label.strip()) <= 120 for label in labels)):
            raise ModError("Schema enumLabels must name every enum value in order")
    json_bytes(schema)


def validate_value(value, schema, path="value"):
    """Check a finite JSON value against the supported schema; raise ModError on mismatch."""
    json_bytes(value)
    kind = schema["type"]
    valid = {"object": isinstance(value, dict), "array": isinstance(value, list),
             "string": isinstance(value, str), "number": type(value) in (int, float),
             "integer": type(value) is int, "boolean": type(value) is bool,
             "null": value is None}[kind]
    if not valid:
        raise ModError(f"{path} must be {kind}")
    if "enum" in schema and not any(type(value) is type(item) and value == item for item in schema["enum"]):
        raise ModError(f"{path} is outside enum")
    if kind == "object":
        properties = schema.get("properties", {})
        if not set(schema.get("required", [])) <= set(value):
            raise ModError(f"{path} is missing required properties")
        if not schema.get("additionalProperties", False) and set(value) - set(properties):
            raise ModError(f"{path} has unknown properties")
        for key, child in value.items():
            if key in properties:
                validate_value(child, properties[key], f"{path}.{key}")
    elif kind == "array":
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 1000000):
            raise ModError(f"{path} item count is outside limits")
        if "items" in schema:
            for index, item in enumerate(value):
                validate_value(item, schema["items"], f"{path}[{index}]")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 10000000):
            raise ModError(f"{path} length is outside limits")
    elif kind in {"integer", "number"}:
        if value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ModError(f"{path} is outside numeric limits")


def manifest(value, *, builtin=False):
    """Normalize a v1 manifest; executable declarations remain strings and are never imported."""
    allowed = {"apiVersion", "id", "name", "version", "description", "capabilities", "requires",
               "settingsSchema", "settings", "requirements", "license", "provenance", "models", "availability", "defaultEnabled"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ModError("Manifest contains unknown fields")
    result = deepcopy(value)
    if "defaultEnabled" in result and (not builtin or type(result["defaultEnabled"]) is not bool):
        raise ModError("Only builtins may declare boolean defaultEnabled")
    if type(result.get("apiVersion")) is not int or result["apiVersion"] != 1:
        raise ModError("Unsupported apiVersion; expected integer 1")
    identifier(result.get("id"), "mod id")
    for key, limit in (("name", 160), ("version", 80), ("description", 2000)):
        item = result.get(key, "" if key == "description" else None)
        if not isinstance(item, str) or len(item) > limit or (key != "description" and not item):
            raise ModError(f"Invalid manifest {key}")
    if not VERSION.fullmatch(result["version"]):
        raise ModError("Version must use semantic major.minor.patch form")
    capabilities = result.get("capabilities")
    if not isinstance(capabilities, list) or not 1 <= len(capabilities) <= 32:
        raise ModError("Declare between 1 and 32 capabilities")
    seen = set()
    for capability in capabilities:
        if not isinstance(capability, dict) or set(capability) - {"id", "kind", "entrypoint", "inputSchema", "outputSchema", "rawOutputSchema", "contract", "inputContract", "prepare_input", "convert_output", "default", "description"}:
            raise ModError("Invalid capability declaration")
        cap_id = identifier(capability.get("id"), "capability id")
        if cap_id in seen:
            raise ModError("Duplicate capability id")
        seen.add(cap_id)
        capability.setdefault("kind", "provider")
        if capability["kind"] not in {"provider", "stage"}:
            raise ModError("Capability kind must be provider or stage")
        if "default" in capability and (type(capability["default"]) is not bool or not builtin):
            raise ModError("Only bundled capabilities may declare a default")
        if not builtin or "entrypoint" in capability:
            entrypoint(capability.get("entrypoint", ""))
        for key in ("prepare_input", "convert_output"):
            if key in capability:
                if builtin:
                    raise ModError("Format adapter hooks belong to user JSON workers, not opaque builtin callbacks")
                entrypoint(capability[key])
        for key, direction in (("contract", "output"), ("inputContract", "input")):
            if key in capability:
                resolve_contract(capability[key], capability=cap_id, direction=direction)
        for key in ("inputSchema", "outputSchema", "rawOutputSchema"):
            if key in capability:
                validate_schema(capability[key])
    required = result.setdefault("requires", [])
    if (not isinstance(required, list) or len(required) > 16
            or any(not isinstance(item, str) for item in required) or len(set(required)) != len(required)):
        raise ModError("requires must list unique mod IDs")
    for dependency in required:
        identifier(dependency, "dependency id")
        if dependency == result["id"]:
            raise ModError("A mod cannot require itself")
    settings_schema = result.setdefault("settingsSchema", {"type": "object", "properties": {}, "additionalProperties": False})
    validate_schema(settings_schema)
    if settings_schema["type"] != "object":
        raise ModError("Settings schema must describe an object")
    validate_value(result.setdefault("settings", {}), settings_schema, "settings defaults")
    requirements = result.setdefault("requirements", {})
    if not isinstance(requirements, dict) or set(requirements) - {"python", "packages", "files", "platforms"}:
        raise ModError("Invalid requirements declaration")
    if "python" in requirements and (not isinstance(requirements["python"], str) or not re.fullmatch(r">=\d+\.\d+", requirements["python"])):
        raise ModError("Python requirement must be >=major.minor")
    for key in ("packages", "files", "platforms"):
        values = requirements.get(key, [])
        if not isinstance(values, list) or len(values) > 64 or any(not isinstance(item, str) or not item or len(item) > 1000 for item in values):
            raise ModError(f"Invalid requirements.{key}")
    for key in ("license", "provenance"):
        if key not in result or not isinstance(result[key], (dict, str)) or not result[key]:
            raise ModError(f"Manifest requires attributable {key}")
    if "models" in result and (not isinstance(result["models"], list) or len(result["models"]) > 64):
        raise ModError("models must be a bounded metadata list")
    if len(json_bytes(result)) > 65536:
        raise ModError("Manifest exceeds 64 KiB")
    return result
