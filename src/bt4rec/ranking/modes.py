from __future__ import annotations

from typing import Any, Dict, List

from bt4rec.inference.constants import AB_MODES
from bt4rec.ranking.prompts import build_native_prompt, controlled_prompt


def requested_modes(raw: str) -> List[str]:
    modes = [s.strip().lower() for s in raw.split(',') if s.strip()]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('--modes cannot be empty or contain duplicates')
    unknown = sorted(set(modes) - set(AB_MODES))
    if unknown:
        raise ValueError(f'Unknown modes: {unknown}. Allowed: {list(AB_MODES)}')
    return modes


def mode_prompt(mode: str, item: Dict[str, Any]) -> str:
    """Filled-in ranking prompt of one condition (see bt4rec.inference.constants)."""
    h, c, latest, hows = item['history_items'], item['candidate_items'], item['latest_behavior'], item['hows']
    if mode == 'native':
        # Plain ranking prompt: no CoMemTree WHAT/HOW sections or rules.
        return build_native_prompt(history_items=h, candidate_items=c)
    if mode == 'latest_what_only':
        # Native prompt + a short summary of the latest window (WHAT).
        return build_native_prompt(history_items=h, candidate_items=c, latest_behavior=latest)
    if mode == 'latest_how_only':
        return controlled_prompt(h, c, None, hows['latest_how'])
    if mode == 'tree':
        return controlled_prompt(h, c, latest, hows['tree'])
    if mode == 'tree_v2':
        return controlled_prompt(h, c, latest, hows['tree'], recent3=item['recent3_behaviors'])
    if mode == 'random_how_only':
        return controlled_prompt(h, c, None, hows['random_how'])
    if mode in ('global_continuation', 'pmn', 'oracle'):
        return controlled_prompt(h, c, latest, hows[mode])
    raise ValueError(f'Unknown mode {mode!r}')
