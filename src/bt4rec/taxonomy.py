"""Single source of truth for HOW-label normalisation and state-mapping text.

Before the refactor, the Tree builder and the inference scripts each carried their
own copy of normalize_mechanism/scope/direction. They disagreed on punctuation:
the builder kept '/', so ``same collection/series`` stayed as is during clustering
but became ``same series`` at inference, which changed the embedded
``structured_combined`` text for the same behavior. Every module now imports the
functions below, so training and test always produce the same text.

TAXONOMY_VERSION is written into Tree metadata. Trees built with the legacy
builder (no version) should be rebuilt to get exact train/test agreement.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

from bt4rec.common.text import clean_text

TAXONOMY_VERSION = "bt4rec_taxonomy_v1_punctuation_insensitive"

# Allowed labels exactly as written in the behaviour-extraction prompts.
MECHANISMS: Tuple[str, ...] = (
    "repetition", "persistence", "collection expansion", "deepening", "narrowing",
    "broadening", "shifting", "returning", "adjacent exploration",
    "cross-category exploration", "refinement", "unknown",
)
SCOPES: Tuple[str, ...] = (
    "same item", "same creator", "same collection/series", "same subcategory",
    "same broad category", "related category", "cross-category", "mixed", "unknown",
)
DIRECTIONS: Tuple[str, ...] = (
    "stable", "repeat", "deepen", "narrow", "broaden", "shift", "return", "mixed", "unknown",
)

MECHANISM_ALIASES = {
    "switching": "shifting",
    "shift": "shifting",
    "collection completion": "collection expansion",
    "preference deepening": "deepening",
    "preference refinement": "refinement",
    "repeat": "repetition",
}
SCOPE_ALIASES = {
    "same artist": "same creator",
    "same artists": "same creator",
    "same collection series": "same series",
    "same collection": "same series",
    "same series collection": "same series",
    "same broad category": "same category",
    "same broad categories": "same category",
    "same categories": "same category",
    "related category": "adjacent category",
}
DIRECTION_ALIASES = {
    "switch": "shift",
    "shifting": "shift",
    "deepening": "deepen",
    "broadening": "broaden",
    "narrowing": "narrow",
    "repetition": "repeat",
    "returning": "return",
}


def normalize_label(x: Any) -> str:
    """Lower-case and treat every non-alphanumeric character as a space."""
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return re.sub(r"\s+", " ", s).strip() or "unknown"


def normalize_mechanism(x: Any) -> str:
    s = normalize_label(x)
    return MECHANISM_ALIASES.get(s, s)


def normalize_scope(x: Any) -> str:
    s = normalize_label(x)
    return SCOPE_ALIASES.get(s, s)


def normalize_direction(x: Any) -> str:
    s = normalize_label(x)
    return DIRECTION_ALIASES.get(s, s)


def normalize_choice(value: Any, allowed: Sequence[str], field: str) -> str:
    """Map a model label onto one allowed prompt label (strict; raises if none).

    Fallback for the external ``oracle_probe_qwen.normalize_choice`` used by the
    Qwen precompute; that module is preferred when it is importable.
    """
    norm = {
        "mechanism": normalize_mechanism,
        "scope": normalize_scope,
        "direction": normalize_direction,
    }.get(field, normalize_label)
    target = norm(value)
    for label in allowed:
        if norm(label) == target:
            return label
    raise ValueError(f"{field}: {value!r} is not one of {list(allowed)}")


# -----------------------------------------------------------------------------
# Profiles / clustering keys / text used to embed a behaviour
# -----------------------------------------------------------------------------

def behavior_profile(row: Dict[str, Any]) -> Dict[str, str]:
    return {
        "mechanism": normalize_mechanism(row.get("mechanism")),
        "scope": normalize_scope(row.get("scope")),
        "direction": normalize_direction(row.get("direction")),
    }


def behavior_bucket_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    """Full structured profile retained for diagnostics/state summaries."""
    p = behavior_profile(row)
    return (p["mechanism"], p["scope"], p["direction"])


def clustering_bucket_key(row: Dict[str, Any], constraint_level: str) -> Tuple[str, ...]:
    """Hard compatibility key used by constrained KMeans AND by test-time mapping."""
    p = behavior_profile(row)
    if constraint_level == "mechanism":
        return (p["mechanism"],)
    if constraint_level == "mechanism_scope":
        return (p["mechanism"], p["scope"])
    if constraint_level == "full":
        return (p["mechanism"], p["scope"], p["direction"])
    raise ValueError(f"Unknown constraint_level={constraint_level}")


def normalize_bucket_key(key: Sequence[str], constraint_level: str) -> Tuple[str, ...]:
    """Re-normalise a bucket key stored in a (possibly legacy) behavior_states.json."""
    fns = (normalize_mechanism, normalize_scope, normalize_direction)
    n = {"mechanism": 1, "mechanism_scope": 2, "full": 3}[constraint_level]
    return tuple(fn(v) for fn, v in zip(fns[:n], list(key)[:n]))


def abstract_behavior_text(profile: Dict[str, str]) -> str:
    return (
        f"mechanism={profile['mechanism']}; "
        f"scope={profile['scope']}; "
        f"direction={profile['direction']}"
    )


def get_trajectory_signature(row: Dict[str, Any]) -> str:
    # New schema uses trajectory_signature. behavior_signature is an alias.
    return clean_text(row.get("trajectory_signature") or row.get("behavior_signature"))


def choose_cluster_text(row: Dict[str, Any], field: str) -> str:
    """Text embedded for KMeans (builder) and for state mapping (inference).

    recommendation_behavior / semantic_focus NEVER enter this text.
    """
    signature = get_trajectory_signature(row)
    p = behavior_profile(row)
    if field in {"trajectory_signature", "behavior_signature"}:
        text = signature
    elif field == "structured_combined":
        text = (
            f"mechanism={p['mechanism']}; "
            f"scope={p['scope']}; "
            f"direction={p['direction']}; "
            f"trajectory={signature}"
        )
    else:
        raise ValueError(f"Unsupported cluster_text_field={field!r}")
    return clean_text(text)


def compatible_key_for_state(state_row: Dict[str, Any], constraint_level: str) -> Optional[Tuple[str, ...]]:
    """Bucket key of a frozen state, falling back to its majority profile."""
    key = state_row.get("behavior_bucket_key")
    if isinstance(key, (list, tuple)) and key:
        return normalize_bucket_key(key, constraint_level)
    return clustering_bucket_key(state_row, constraint_level)
