from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple

from bt4rec.common.text import extract_json_value


def sanitize_ranking(
    raw_output: Any,
    valid_ids: Sequence[str],
) -> Tuple[
    List[str],
    bool,
    int,
    bool,
    str,
    List[int],
]:
    """
    Parse C01..Cxx from the FIRST generated ranking line and map them back to
    actual item IDs.

    Policy shared with the A-LLMRec ranker:
    - keep the FIRST occurrence of each valid candidate label;
    - ignore duplicate labels and out-of-range labels;
    - append missing labels in the ORIGINAL frozen candidate order;
    - only fall back to the full frozen candidate order when no valid Cxx label
      can be extracted at all.

    ``parse_complete`` is stricter than ``parse_ok``: it is True only when the
    model itself returned every candidate exactly once with no invalid labels.
    """
    valid = [
        str(x)
        for x in valid_ids
    ]
    n = len(valid)
    text = str(
        raw_output
        if raw_output is not None
        else ""
    ).strip()

    if not text:
        return (
            list(valid),
            False,
            0,
            False,
            "empty_output",
            [],
        )

    # Parse the JSON list only. Labels mentioned in reasoning are never ranking items.
    try:
        obj = extract_json_value(text)
    except (ValueError, TypeError):
        # Recover the common invalid apostrophe escape without inventing structure.
        try:
            obj = extract_json_value(text.replace("\\'", "'"))
        except (ValueError, TypeError):
            return list(valid), False, 0, False, "invalid_json", []
    values = obj.get("ranking", []) if isinstance(obj, dict) else obj
    if not isinstance(values, list):
        return list(valid), False, 0, False, "ranking_not_list", []
    nums = []
    for value in values:
        match = re.fullmatch(r"C(\d{2,3})", value) if isinstance(value, str) else None
        nums.append(int(match.group(1)) if match else 0)

    seen = set()
    parsed_order: List[int] = []
    duplicate_found = False
    invalid_found = False

    for x in nums:
        if not (
            1 <= x <= n
        ):
            invalid_found = True
            continue
        if x in seen:
            duplicate_found = True
            continue
        parsed_order.append(x)
        seen.add(x)

    returned_n = len(
        parsed_order
    )

    # True parse failure: no usable candidate label was returned.
    if not parsed_order:
        return (
            list(valid),
            False,
            0,
            False,
            "no_valid_candidate_labels",
            [],
        )

    missing = [
        x
        for x in range(
            1,
            n + 1,
        )
        if x not in seen
    ]

    complete = bool(
        returned_n == n
        and not duplicate_found
        and not invalid_found
        and not missing
    )

    repaired_order = (
        parsed_order
        + missing
    )
    ranking = [
        valid[x - 1]
        for x in repaired_order
    ]

    reasons: List[str] = []
    if duplicate_found:
        reasons.append(
            "duplicate"
        )
    if invalid_found:
        reasons.append(
            "invalid"
        )
    if missing:
        reasons.append(
            "missing"
        )

    parse_reason = (
        "ok_complete_ranking"
        if complete
        else (
            "repaired_"
            + "_and_".join(
                reasons
            )
        )
    )

    assert len(ranking) == n
    assert len(set(ranking)) == n
    assert set(ranking) == set(valid)

    return (
        ranking,
        complete,
        returned_n,
        True,
        parse_reason,
        repaired_order,
    )
