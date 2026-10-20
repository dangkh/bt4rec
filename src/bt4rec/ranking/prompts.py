from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bt4rec.common.text import clean_text


def format_history(
    items: List[Dict[str, str]],
) -> str:
    return "\n".join(
        f"{i}. {x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def format_candidates(
    items: List[Dict[str, str]],
) -> str:
    """
    Candidate labels are an output interface only.  Raw item IDs are intentionally
    hidden from the ranking prompt so Qwen never has to copy Amazon/MovieLens IDs.
    """
    return "\n".join(
        f"[C{i:02d}] {x['title']} | {x['category']}"
        for i, x in enumerate(
            items,
            start=1,
        )
    )


def build_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    latest_behavior: Optional[Dict[str, Any]],
    predicted_transition: Optional[
        List[Dict[str, Any]]
    ],
) -> str:
    """
    ONE controlled template for Native / Latest-only / Tree.

    The only difference across conditions is which evidence blocks are populated.
    Candidate labels C01..Cxx are used only as stable ranking handles.
    """
    n = len(candidate_items)

    latest_block = (
        "Not provided."
    )
    if latest_behavior is not None:
        latest_block = json.dumps(
            {
                "recommendation_behavior": (
                    latest_behavior.get(
                        "recommendation_behavior",
                        "",
                    )
                ),
                "semantic_focus": (
                    latest_behavior.get(
                        "semantic_focus",
                        [],
                    )
                ),
            },
            ensure_ascii=False,
            indent=2,
        )

    transition_block = (
        "Not provided."
    )
    if predicted_transition:
        transition_block = json.dumps(
            predicted_transition,
            ensure_ascii=False,
            indent=2,
        )

    if n == 20:
        example = (
            "[C07, C03, C12, C01, C18, C05, C09, C14, C02, C20, "
            "C11, C06, C16, C04, C13, C08, C19, C10, C15, C17]"
        )
    else:
        example_order = (
            list(range(2, n + 1, 2))
            + list(range(1, n + 1, 2))
        )
        example = (
            "["
            + ", ".join(
                f"C{x:02d}"
                for x in example_order
            )
            + "]"
        )

    return f"""You are ranking candidate items for the user's NEXT choice.

AVAILABLE SIGNALS

1) RECENT HISTORY
This is the user's observed item history.

{format_history(history_items)}

2) LATEST OBSERVED BEHAVIOR — WHAT the user currently prefers
{latest_block}

3) PREDICTED NEXT TRANSITION — HOW the current preference is expected to evolve
{transition_block}

INTERPRETATION RULES
- Use only the signals that are provided.
- The LATEST OBSERVED BEHAVIOR is the semantic anchor: it describes WHAT content
  the target user currently prefers.
- If a PREDICTED NEXT TRANSITION is provided, interpret it RELATIVE TO the latest
  observed behavior. It describes HOW the current preference is expected to evolve,
  not a specific content identity.
- Do NOT replace the target user's semantic preference with content from another user.
- If multiple predicted transitions are provided, earlier entries have higher
  collaborative probability and should receive more weight.
- Use recent history as supporting context and tie-breaking evidence.
- Candidate input order is RANDOM and has no ranking meaning.
- Base matching only on information actually shown in history, behavior, transition,
  title, and category.
- CANDIDATE LABELS (C01, C02, ...) are arbitrary handles and carry no preference signal.

CANDIDATE ITEMS

{format_candidates(candidate_items)}

DECISION PROCEDURE
1. Infer the target user's current semantic preference from the available personal evidence.
2. If a predicted transition is provided, apply its mechanism/scope/trajectory to that
   current semantic preference.
3. Rank candidates that best instantiate the resulting expected NEXT preference.
4. Rank all remaining candidates by decreasing compatibility.

OUTPUT REQUIREMENTS
- Rank ALL {n} candidate labels exactly once.
- Use ONLY C01, C02, ..., C{n:02d}.
- Do NOT output bare integers such as 1, 2, 3.
- Do NOT repeat any candidate label.
- Do NOT omit any candidate label.
- Return one JSON object with "ranking" and "reasoning" fields.
- "ranking" must be a list of exactly {n} quoted candidate labels, best first.
- "reasoning" must be one short sentence (at most 25 words).
- Do not output item IDs, item titles in the ranking list, code fences or a second answer.
- Apostrophes do not need escaping in JSON strings.
Return ALL labels; do not stop after the top few candidates.
"""


def build_recent3_rank_prompt(
    *,
    history_items: List[Dict[str, str]],
    candidate_items: List[Dict[str, str]],
    recent3_behaviors: List[Dict[str, Any]],
) -> str:
    """Recent-3 baseline: three observed semantic behaviors, NO Tree transition.

    Build from the *unchanged* List ranking prompt, then replace only the
    observed-behavior section and its interpretation. The existing Native,
    Latest-only, and Tree prompts are never modified.
    """
    if not recent3_behaviors or len(recent3_behaviors) > 3:
        raise ValueError("Recent-3 requires between 1 and 3 observed behaviors")
    prompt = build_rank_prompt(
        history_items=history_items,
        candidate_items=candidate_items,
        latest_behavior=None,
        predicted_transition=None,
    )
    semantic_evidence = [
        {
            "recommendation_behavior": b.get("recommendation_behavior", ""),
            "semantic_focus": b.get("semantic_focus", []),
        }
        for b in recent3_behaviors
    ]
    recent_block = json.dumps(semantic_evidence, ensure_ascii=False, indent=2)
    old_section = "2) LATEST OBSERVED BEHAVIOR — WHAT the user currently prefers\nNot provided."
    new_section = (
        "2) RECENT OBSERVED BEHAVIORS — WHAT the user has preferred "
        "across recent local windows (oldest to newest)\n"
        + recent_block
    )
    if prompt.count(old_section) != 1:
        raise AssertionError("Recent-3 prompt failed to locate the observed-behavior section")
    prompt = prompt.replace(old_section, new_section, 1)
    old_rule = (
        "- The LATEST OBSERVED BEHAVIOR is the semantic anchor: it describes WHAT content\n"
        "  the target user currently prefers."
    )
    new_rule = (
        "- The RECENT OBSERVED BEHAVIORS are chronological personal semantic evidence.\n"
        "  Consider all available windows; emphasize newer evidence without ignoring\n"
        "  recurring patterns. These are observations, NOT predicted transitions."
    )
    if prompt.count(old_rule) != 1:
        raise AssertionError("Recent-3 prompt failed to locate the interpretation rule")
    prompt = prompt.replace(old_rule, new_rule, 1)
    # The transition section remains 'Not provided.' and no Tree-derived state
    # evidence is included. All ranking and output-format instructions stay intact.
    return prompt


def controlled_prompt(history_items, candidate_items, latest, how, recent3=None):
    """ALL controlled conditions use identical guidance framing and output rules.

    The original bt4rec prompt is reused verbatim, apart from the minimal,
    mode-independent changes that remove the claim that even random/observed HOW
    is a genuinely PREDICTED transition. Only the HOW block varies by condition.
    """
    if recent3 is None:
        prompt = build_rank_prompt(history_items=history_items, candidate_items=candidate_items,
                                   latest_behavior=latest, predicted_transition=None)
    else:
        prompt = build_recent3_rank_prompt(history_items=history_items,
                                           candidate_items=candidate_items,
                                           recent3_behaviors=recent3)
    old = '3) PREDICTED NEXT TRANSITION — HOW the current preference is expected to evolve\nNot provided.'
    val = (json.dumps(how, ensure_ascii=False, indent=2) if how is not None else 'Not provided.')
    new = '3) ADDITIONAL BEHAVIOR GUIDANCE — possible HOW evolution of the current preference\n' + val
    if prompt.count(old) != 1:
        raise RuntimeError('bt4rec base ranking template changed; cannot build controlled prompt')
    prompt = prompt.replace(old, new, 1)
    prompt = prompt.replace('If a PREDICTED NEXT TRANSITION is provided,',
                            'If ADDITIONAL BEHAVIOR GUIDANCE is provided,')
    prompt = prompt.replace('If multiple predicted transitions are provided, earlier entries have higher\n'
                            '  collaborative probability and should receive more weight.\n', '')
    prompt = prompt.replace('If a predicted transition is provided, apply its mechanism/scope/trajectory to that',
                            'If additional behavior guidance is provided, apply its mechanism/scope/trajectory to that')
    # Prompt has the same rule for source and Tree, and does not identify condition.
    assert 'PREDICTED NEXT TRANSITION' not in prompt
    assert 'higher collaborative probability' not in prompt
    return prompt


def append_order1_persistence_prompt_v1(prompt: str, persistence: Dict[str, Any]) -> str:
    """Legacy (v1) block: one scalar, verbose wording. Kept for reproducing old runs."""
    header = '4) LAST-BEHAVIOR PERSISTENCE — order-1, last observed HOW state only'
    if persistence.get('available'):
        value = f"{float(persistence['probability_next_same_behavior_state']):.4f}"
        block = (
            header + '\n'
            + 'P(next behavior state = latest behavior state | latest behavior state) = ' + value + '\n'
            + 'This estimates persistence of the same abstract HOW pattern, not repetition of an item.\n'
            + 'The estimate uses only the latest behavior state (order-1), independent of the\n'
            + 'longer-context next-behavior prediction above. Treat it as a soft calibration\n'
            + 'signal for continuity versus change, not as a guaranteed outcome.\n'
        )
    else:
        block = (
            header + '\n'
            + 'Not estimable from the frozen Tree for this user; do not assume zero\n'
            + 'persistence or substitute a ROOT-only probability.\n'
        )
    anchor = 'CANDIDATE ITEMS\n'
    if prompt.count(anchor) != 1:
        raise AssertionError('Expected exactly one candidate block in bt4rec prompt')
    return prompt.replace(anchor, block + '\n' + anchor, 1)


def append_order1_persistence_prompt(
    prompt: str,
    persistence: Dict[str, Any],
    latest_behavior: Optional[Dict[str, Any]] = None,
    version: str = "v2",
    latest_how: Optional[Dict[str, Any]] = None,
) -> str:
    """Section 4: probability that the next behavior keeps the latest behavior pattern.

    v2 (default) names the latest observed behavior (WHAT) the probability refers to,
    and its observed HOW (mechanism / scope / trajectory of the user's own latest
    window), so the probability is anchored to a concrete behavior pattern.
    """
    if version == "v1":
        return append_order1_persistence_prompt_v1(prompt, persistence)
    if version != "v2":
        raise ValueError(f"Unknown stay prompt version {version!r}")
    lines = ["4) LAST-BEHAVIOR PERSISTENCE"]
    if latest_behavior:
        what = clean_text(latest_behavior.get("recommendation_behavior"))
        focus = [clean_text(x) for x in (latest_behavior.get("semantic_focus") or []) if clean_text(x)]
        if what:
            lines.append("Latest observed behavior: " + what)
        if focus:
            lines.append("Latest semantic focus: " + ", ".join(focus))
    if latest_how:
        parts = []
        if clean_text(latest_how.get("mechanism")):
            parts.append("mechanism=" + clean_text(latest_how["mechanism"]))
        if clean_text(latest_how.get("scope")):
            parts.append("scope=" + clean_text(latest_how["scope"]))
        how = ", ".join(parts)
        traj = clean_text(latest_how.get("trajectory_signature"))
        if traj:
            how = (how + "; " if how else "") + "trajectory: " + traj
        if how:
            lines.append("Latest observed HOW: " + how)
    if persistence.get("available"):
        value = f"{float(persistence['probability_next_same_behavior_state']):.4f}"
        lines.append("Probability that the next behavior stays with this behavior pattern: " + value)
    else:
        lines.append("Probability that the next behavior stays with this behavior pattern: not available.")
    lines.append("This estimates persistence of the same abstract HOW pattern, not repetition of an item.")
    block = "\n".join(lines) + "\n"
    anchor = "CANDIDATE ITEMS\n"
    if prompt.count(anchor) != 1:
        raise AssertionError("Expected exactly one candidate block in bt4rec prompt")
    return prompt.replace(anchor, block + "\n" + anchor, 1)
