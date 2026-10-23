"""Optional i2Agent-style SCORE output for the CoMemTree ranking prompt (--rank-format score).

Only the OUTPUT REQUIREMENTS block (and the system prompt) change: every evidence
section above it (history, WHAT, HOW, P(stay), candidates, rules) is byte-identical
to the default list prompt. Output and repair follow baseline/i2agent.py exactly:

* the model returns {"scores": {"C01": 0.92, ...}} with one score in [0, 1] per label;
* labels are normalised leniently (c3, C3, [C03], C003 -> C03);
* duplicates keep the first value; invalid labels / non-numeric / out-of-range scores
  are dropped;
* ranking = valid labels sorted by score (desc), ties in frozen candidate order;
* missing labels are appended at the end in frozen candidate order;
* no valid score at all -> frozen candidate order (fallback). No retry.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Tuple

from bt4rec.common.text import extract_json_value

SCORE_FORMAT_VERSION = "i2agent_score_output_v1"
SCORE_SYSTEM_PROMPT = "You are a recommendation ranking system. Return only the requested valid JSON."
_ANCHOR = "OUTPUT REQUIREMENTS\n"


def _labels(n: int) -> List[str]:
    w = max(2, len(str(max(1, n))))
    return [f"C{i:0{w}d}" for i in range(1, n + 1)]


def score_output_block(n: int) -> str:
    """Identical wording to baseline/i2agent.py::build_score_prompt (OUTPUT REQUIREMENTS)."""
    return (
        "OUTPUT REQUIREMENTS\n"
        "- Assign each candidate one relevance score in [0, 1]; larger means more relevant.\n"
        f"- Include ALL {n} candidate labels exactly once: {', '.join(_labels(n))}.\n"
        "- Do not invent, omit, or duplicate labels. Never output raw item IDs.\n"
        "- Scores do not need to sum to 1; avoid assigning identical scores unless relevance "
        "is genuinely indistinguishable.\n"
        "- Return ONLY a JSON object containing a scores dictionary. No explanation or markdown.\n"
        "- Silently verify every required label appears exactly once before returning.\n"
        "\n"
        '{"scores":{"C01":0.92,"C02":0.37}}\n'
    )


def to_score_prompt(prompt: str, n: int) -> str:
    """Replace the list OUTPUT REQUIREMENTS block (always the last section) by the score block."""
    if prompt.count(_ANCHOR) != 1:
        raise RuntimeError("ranking prompt must contain exactly one OUTPUT REQUIREMENTS block")
    head = prompt[: prompt.index(_ANCHOR)]
    head = head.replace("4. Rank all remaining candidates by decreasing compatibility.",
                        "4. Score every candidate independently for relevance to the user's NEXT choice.")
    return head + score_output_block(n)


def normalize_candidate_label(value: Any, n: int):
    label = str(value).strip().upper().strip("[](){}<>")
    m = re.fullmatch(r"C0*(\d+)", label)
    if m is None or not (1 <= int(m.group(1)) <= n):
        return None
    return f"C{int(m.group(1)):0{max(2, len(str(n)))}d}"


def parse_score_ranking(raw: Any, valid_ids: Sequence[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Same parser/repair as i2Agent; meta keys compatible with the list parser."""
    valid = [str(x) for x in valid_ids]
    n = len(valid)
    labels = _labels(n)
    label_to_id = dict(zip(labels, valid))
    scores: Dict[str, float] = {}
    invalid_labels: List[str] = []
    duplicate_labels: List[str] = []
    invalid_scores: List[str] = []
    json_ok, parse_error = False, None
    try:
        obj = extract_json_value(str(raw or ""))
        if not isinstance(obj, dict) or "scores" not in obj:
            raise ValueError('Expected JSON object with "scores" field')
        values = obj["scores"]
        if isinstance(values, dict):
            pairs = list(values.items())
        elif isinstance(values, list):
            pairs = []
            for entry in values:
                if not isinstance(entry, dict):
                    invalid_scores.append(str(entry))
                    continue
                pairs.append((entry.get("label", ""), entry.get("score")))
        else:
            raise ValueError('"scores" must be dictionary or list of {label,score}')
        json_ok = True
        for key, value in pairs:
            label = normalize_candidate_label(key, n)
            if label not in label_to_id:
                invalid_labels.append(str(key))
                continue
            if label in scores:
                duplicate_labels.append(label)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                invalid_scores.append(f"{label}={value!r}")
                continue
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                invalid_scores.append(f"{label}={value!r}")
                continue
            scores[label] = float(value)
    except Exception as exc:  # noqa: BLE001 - same policy as i2Agent
        parse_error = str(exc)
    missing = [lab for lab in labels if lab not in scores]
    ranked = sorted(scores, key=lambda lab: -scores[lab])   # stable: ties keep frozen order
    ranked.extend(missing)
    ranking = [label_to_id[lab] for lab in ranked]
    complete = bool(json_ok and not missing and not duplicate_labels
                    and not invalid_labels and not invalid_scores)
    reasons = [r for r, bad in (("duplicate", duplicate_labels), ("invalid", invalid_labels or invalid_scores),
                                ("missing", missing)) if bad]
    if not scores:
        reason = "invalid_json" if not json_ok else "no_valid_candidate_scores"
    else:
        reason = "ok_complete_ranking" if complete else "repaired_" + "_and_".join(reasons)
    order = [labels.index(lab) + 1 for lab in ranked]
    return ranking, {
        "rank_format": "score",
        "score_format_version": SCORE_FORMAT_VERSION,
        "parse_ok": bool(scores),
        "parse_json_ok": json_ok,
        "parse_complete": complete,
        "parse_reason": reason,
        "model_returned_valid_ids": len(scores),
        "num_candidates": n,
        "parsed_candidate_order_1based": order,
        "parsed_candidate_labels": ranked,
        "scores_by_label": scores,
        "score_tie_count": len(scores) - len(set(scores.values())),
        "missing_labels": missing,
        "duplicate_labels": duplicate_labels,
        "invalid_labels": invalid_labels,
        "invalid_scores": invalid_scores,
        "parse_error": parse_error,
        "repair_used": bool(scores) and not complete,
        "fallback_used": not bool(scores),
        "reasoning": "",
        "raw_output": str(raw or ""),
    }
