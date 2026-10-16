from __future__ import annotations

import json
import re
from typing import Any


def clean_text(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip())


def extract_json_value(text: str) -> Any:
    """Recover the first complete JSON object/array from model output."""
    s = str(text or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    starts = []
    for opener, closer in [("{", "}"), ("[", "]")]:
        pos = s.find(opener)
        if pos >= 0:
            starts.append((pos, opener, closer))

    if not starts:
        raise ValueError("No JSON value found")

    start, opener, closer = min(starts, key=lambda z: z[0])
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i + 1], strict=False)

    raise ValueError("Unbalanced JSON")
