"""Field-level scoring for extraction evals."""
from __future__ import annotations

import re
from collections import Counter
from itertools import combinations

SCREENSHOT_FIELDS = ["report_type", "species", "pet_name", "area", "event_date",
                     "contact_phone", "colors", "has_pet_photo"]


def _norm(s) -> str | None:
    if s is None:
        return None
    s = re.sub(r"[\"'`׳״.,!\s]+", " ", str(s)).strip().lower()
    return s or None


def digits(s) -> str | None:
    if s is None:
        return None
    d = re.sub(r"\D", "", str(s))
    if d.startswith("972"):  # +972 50... -> 050...
        d = "0" + d[3:]
    return d or None


def score_screenshot(truth: dict, pred: dict) -> dict[str, bool | None]:
    """True/False per field; None = not scored for this item."""
    s: dict[str, bool | None] = {}
    s["report_type"] = str(pred.get("report_type", "")).lower() == truth["report_type"]
    s["species"] = str(pred.get("species", "")).lower() == truth["species"]
    s["pet_name"] = _norm(pred.get("pet_name")) == _norm(truth["pet_name"])
    # area: the city must be there; neighborhood wording may vary
    area = _norm(pred.get("area")) or ""
    s["area"] = _norm(truth["city"]) in area
    s["event_date"] = pred.get("event_date") == truth["event_date"]
    s["contact_phone"] = digits(pred.get("contact_phone")) == truth["contact_phone"]
    if truth["colors"] is None:
        s["colors"] = None
    else:
        s["colors"] = {c.lower() for c in pred.get("primary_colors", [])} == set(truth["colors"])
    s["has_pet_photo"] = bool(pred.get("has_pet_photo")) == truth["has_pet_photo"]
    return s


def field_accuracy(scores: list[dict]) -> dict[str, float]:
    out = {}
    for f in SCREENSHOT_FIELDS:
        vals = [s[f] for s in scores if s.get(f) is not None]
        out[f] = sum(vals) / len(vals) if vals else float("nan")
    out["all_fields"] = sum(all(v for v in s.values() if v is not None) for s in scores) / max(len(scores), 1)
    return out


def consistency(descriptions: list[dict]) -> dict[str, float]:
    """How much descriptions of DIFFERENT photos of the SAME pet agree (1.0 = always the same).

    No labels needed: a description that changes between photos of one dog can't
    help match that dog across a lost and a found report.
    """
    if len(descriptions) < 2:
        return {}
    pairs = list(combinations(descriptions, 2))

    def agree(key):
        return sum(str(a[key]).lower() == str(b[key]).lower() for a, b in pairs) / len(pairs)

    def jaccard(a, b):
        x, y = set(a["primary_colors"]), set(b["primary_colors"])
        return len(x & y) / len(x | y) if x | y else 1.0

    first = [d["primary_colors"][0] if d["primary_colors"] else None for d in descriptions]
    return {
        "species": agree("species"),
        "size": agree("size"),
        "collar": agree("collar"),
        "main_color": Counter(first).most_common(1)[0][1] / len(first),
        "color_jaccard": sum(jaccard(a, b) for a, b in pairs) / len(pairs),
    }
