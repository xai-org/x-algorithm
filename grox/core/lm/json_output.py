import json
import re
from typing import Any

_TAGGED = re.compile(r"(.*)<json>(.*)</json>", re.DOTALL)
_FENCED = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)
_PLACEHOLDERS = frozenset({"n/a", "na", "none", "null", "unknown", "other", "-"})


def extract_json(output: str) -> Any:
    tagged = _TAGGED.search(output)
    text = tagged.group(2).strip() if tagged else output.strip()
    fenced = _FENCED.search(text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def is_placeholder(value: str | None) -> bool:
    if not value:
        return True
    return value.strip().lower() in _PLACEHOLDERS
