from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple


def enforce_structural_consistency(
    mechanism: str,
    scope: str,
    direction: str,
    interaction_sequence: List[Dict[str, Any]],
) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
    """
    Deterministic guardrail for labels with literal structural meaning.

    We do NOT try to infer deepening/collection-expansion automatically.
    We only prevent impossible repetition/same-item/repeat labels when no
    exact item_id repeats. This keeps semantic interpretation with Gemma while
    protecting the Tree from taxonomy collapse.
    """
    item_ids = [
        str(x.get("item_id", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_id", "")).strip()
    ]
    unique_ids = set(item_ids)
    has_exact_repeat = (
        len(item_ids) >= 2
        and len(unique_ids) < len(item_ids)
    )
    all_same_item = (
        len(item_ids) >= 2
        and len(unique_ids) == 1
    )

    categories = [
        str(x.get("item_category", "")).strip()
        for x in interaction_sequence
        if str(x.get("item_category", "")).strip()
        and str(x.get("item_category", "")).strip().lower() != "unknown"
    ]
    same_known_category = (
        len(categories) >= 2
        and len(set(categories)) == 1
    )

    adjustments: List[str] = []

    mechanism = str(mechanism or "unknown").strip().lower()
    scope = str(scope or "unknown").strip().lower()
    direction = str(direction or "unknown").strip().lower()

    if not has_exact_repeat:
        if mechanism == "repetition":
            old = mechanism
            # Do not guess persistence/deepening here. The item category can be
            # too broad (e.g., all CDs), so an impossible literal repetition is
            # converted to unknown and exposed for audit.
            mechanism = "unknown"
            adjustments.append(
                f"mechanism:{old}->{mechanism}:no_exact_item_repeat"
            )

        if scope == "same item":
            old = scope
            scope = "mixed"
            adjustments.append(
                f"scope:{old}->{scope}:no_exact_item_repeat"
            )

        if direction == "repeat":
            old = direction
            direction = "unknown"
            adjustments.append(
                f"direction:{old}->{direction}:no_exact_item_repeat"
            )

    # If every observed interaction is literally the same item, these labels
    # have unambiguous structural support.
    if all_same_item:
        if mechanism in {"unknown", "persistence"}:
            adjustments.append(
                f"mechanism:{mechanism}->repetition:all_same_item"
            )
            mechanism = "repetition"
        if scope != "same item":
            adjustments.append(
                f"scope:{scope}->same item:all_same_item"
            )
            scope = "same item"
        if direction in {"unknown", "stable"}:
            adjustments.append(
                f"direction:{direction}->repeat:all_same_item"
            )
            direction = "repeat"

    evidence = {
        "num_items": len(item_ids),
        "num_unique_item_ids": len(unique_ids),
        "has_exact_item_repeat": bool(has_exact_repeat),
        "all_same_item": bool(all_same_item),
        "same_known_category": bool(same_known_category),
    }

    return mechanism, scope, direction, adjustments, evidence


def parse_json_response(text: str) -> Dict[str, Any]:
    text = text.strip()

    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1

    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)
