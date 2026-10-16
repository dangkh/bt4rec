from __future__ import annotations

from typing import Any, Dict, List

from bt4rec.inference.constants import AB_MODES, STAY_MODES
from bt4rec.ranking.prompts import append_order1_persistence_prompt, build_rank_prompt, controlled_prompt


def requested_modes(raw: str) -> List[str]:
    modes = [s.strip().lower() for s in raw.split(',') if s.strip()]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('--modes cannot be empty or contain duplicates')
    unknown = sorted(set(modes) - set(AB_MODES))
    if unknown:
        raise ValueError(f'Unknown ablation modes: {unknown}. Allowed: {AB_MODES}')
    return modes


def mode_prompt(mode: str, item: Dict[str, Any]):
    h,c,latest = item['history_items'],item['candidate_items'],item['latest_behavior']
    if mode == 'tree_original':
        # Byte-for-byte same user-side Tree prompt as original inference_bt4rec.
        return build_rank_prompt(history_items=h, candidate_items=c, latest_behavior=latest,
                                 predicted_transition=item['predicted_transition'])
    if mode in STAY_MODES:
        if mode == 'tree_original_stay':
            base = build_rank_prompt(history_items=h, candidate_items=c,
                                     latest_behavior=latest,
                                     predicted_transition=item['predicted_transition'])
        elif mode == 'recent3_tree_stay':
            base = controlled_prompt(h,c,latest,item['hows']['tree'],recent3=item['recent3_behaviors'])
        elif mode == 'tree_stay':
            base = controlled_prompt(h,c,latest,item['hows']['tree'])
        else:  # stay_only: latest WHAT + persistence, without any next-HOW guidance.
            base = controlled_prompt(h,c,latest,None)
        return append_order1_persistence_prompt(base, item['last_state_persistence'],
                                                latest_behavior=latest,
                                                version=item.get('stay_prompt', 'v2'),
                                                latest_how=item.get('latest_how'))
    if mode == 'native':
        return build_rank_prompt(history_items=h, candidate_items=c,
                                 latest_behavior=None, predicted_transition=None)
    if mode == 'recent3':
        return controlled_prompt(h,c,latest,None,recent3=item['recent3_behaviors'])
    if mode == 'recent3_tree':
        return controlled_prompt(h,c,latest,item['hows']['tree'],recent3=item['recent3_behaviors'])
    return controlled_prompt(h,c,latest,item['hows'].get(mode))
