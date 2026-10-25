"""JSON schemas Claude must fill (structured outputs).

Rules from the API: every object needs "additionalProperties": false, no numeric or
string-length constraints, at most 16 nullable ("x or null") fields per request.
"""
from __future__ import annotations

SPECIES = ["dog", "cat", "other", "unknown"]

# A fixed color vocabulary lets the matcher compare reports without fuzzy text matching.
COLORS = ["black", "white", "brown", "tan", "golden", "cream", "gray",
          "orange", "red", "brindle", "merle", "spotted"]

SIZES = ["small", "medium", "large", "unknown"]

PET_DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "species": {"type": "string", "enum": SPECIES},
        "breed_guess": {"type": ["string", "null"],
                        "description": "Most likely breed or 'mixed', in English; null if unclear"},
        "size": {"type": "string", "enum": SIZES},
        "primary_colors": {"type": "array", "items": {"type": "string", "enum": COLORS},
                           "description": "1-3 main coat colors, most dominant first"},
        "distinctive_markings": {"type": "array", "items": {"type": "string"},
                                 "description": "Short English phrases, e.g. 'white chest patch', 'one ear folded'"},
        "collar": {"type": "string", "enum": ["yes", "no", "not_visible"]},
        "collar_color": {"type": ["string", "null"]},
        "photo_quality": {"type": "string", "enum": ["good", "poor"],
                          "description": "poor = blurry, dark, tiny, or pet mostly hidden"},
    },
    "required": ["species", "breed_guess", "size", "primary_colors", "distinctive_markings",
                 "collar", "collar_color", "photo_quality"],
    "additionalProperties": False,
}

SCREENSHOT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_type": {"type": "string", "enum": ["lost", "found", "sighting", "unclear"],
                        "description": "lost = owner searching; found = someone has the pet; "
                                       "sighting = seen but not caught"},
        "species": {"type": "string", "enum": SPECIES},
        "pet_name": {"type": ["string", "null"], "description": "As written, null if none"},
        "primary_colors": {"type": "array", "items": {"type": "string", "enum": COLORS}},
        "area": {"type": ["string", "null"],
                 "description": "City and neighborhood/street as written in the post (keep Hebrew)"},
        "event_date": {"type": ["string", "null"],
                       "description": "Date lost/found as YYYY-MM-DD. Resolve relative dates "
                                      "('yesterday') using the post's own date if shown; else null"},
        "contact_phone": {"type": ["string", "null"], "description": "Digits only, e.g. 0501234567"},
        "description_he": {"type": ["string", "null"],
                           "description": "The pet's description from the post, in Hebrew, one sentence"},
        "has_pet_photo": {"type": "boolean"},
    },
    "required": ["report_type", "species", "pet_name", "primary_colors", "area", "event_date",
                 "contact_phone", "description_he", "has_pet_photo"],
    "additionalProperties": False,
}

UNSUPPORTED_KEYWORDS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                        "minLength", "maxLength", "maxItems", "pattern"}


def check_schema(schema: dict) -> list[str]:
    """Return problems the API would reject (used by tests)."""
    problems = []

    def walk(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object" and node.get("additionalProperties") is not False:
                problems.append(f"{path}: object without additionalProperties: false")
            if not path.endswith(".properties"):  # keys under "properties" are field names
                for k in sorted(UNSUPPORTED_KEYWORDS & set(node)):
                    problems.append(f"{path}: unsupported keyword {k}")
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(schema, "$")
    nullable = sum(1 for p in schema.get("properties", {}).values()
                   if isinstance(p.get("type"), list) and "null" in p["type"])
    if nullable > 16:
        problems.append(f"{nullable} nullable fields (limit 16)")
    return problems


def validate(obj: dict, schema: dict) -> list[str]:
    """Light check of a response against a flat schema: required keys and enum values."""
    errors = []
    props = schema["properties"]
    for key in schema["required"]:
        if key not in obj:
            errors.append(f"missing {key}")
    for key, value in obj.items():
        spec = props.get(key)
        if spec is None:
            errors.append(f"unexpected {key}")
            continue
        allowed = spec.get("enum") or spec.get("items", {}).get("enum")
        if allowed:
            values = value if isinstance(value, list) else [value]
            bad = [v for v in values if str(v).lower() not in allowed]
            if bad:
                errors.append(f"{key}: {bad} not in vocabulary")
    return errors
