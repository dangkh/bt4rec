from __future__ import annotations

from bt4rec import shared


def prompt_for(last_history: list[dict], future: dict) -> str:
    return f'''You are an analyst of chronological user behavior.
The FINAL item below was chosen by the target user in the future, after the observed history.
Describe the user's observed NEXT preference evolution as an ABSTRACT behavior.

Observed recent item history:
{shared.format_history(last_history)}

Observed future chosen item (ORACLE-ONLY, never available to deployable ranking):
{future['title']} | {future['category']}

Output ONLY JSON with exactly these strings:
{{"mechanism":"...","scope":"...","trajectory_signature":"..."}}
Mechanism: persistence, deepening, adjacent exploration, shifting, collection expansion, or cross category exploration.
Scope: same category, same subcategory, adjacent category, same series, same creator, or cross category.
Trajectory_signature: one short sentence (8-20 words) explaining how preference evolved RELATIVE TO the prior observed items.
Do not output the future item's exact title, names, IDs, labels, candidate rankings, or anything outside the JSON object.'''
