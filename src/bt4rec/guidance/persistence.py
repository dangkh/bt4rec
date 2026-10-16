from __future__ import annotations

from typing import Any, Dict, List, Optional

from bt4rec.inference.constants import STAY_PROMPT_VERSION
from bt4rec.tree.query import ReverseBehaviorTree


def full_weight_probability(node: Dict[str, Any], state_id: str) -> Optional[float]:
    """Return P(state_id) from the *full* stored collaborative weight vector.

    The serialized smoothed_next_distribution may be top-K truncated. In
    particular, it is incorrect to treat a missing state as probability 0.
    """
    weights = node.get('collaborative_next_weights')
    if not isinstance(weights, dict) or not weights:
        return None
    total = sum(float(v) for v in weights.values())
    if total <= 0:
        return None
    return float(weights.get(state_id, 0.0)) / total


def compute_order1_persistence(tree: ReverseBehaviorTree, mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """P(next == B | last B), independent of any longer A,B suffix.

    IMPORTANT: never infer the last state from an older accepted behavior when
    the actual latest window is rejected. Never silently use ROOT as P(.|B).
    Reconstruct the FULL smoothed distribution (B node + ROOT) from weights so
    top-K truncation of serialized rows cannot corrupt the self-probability.
    """
    row = {
        'prompt_version': STAY_PROMPT_VERSION,
        'conditioning': 'last_observed_behavior_state_only',
        'order': 1,
        'available': False,
        'last_state_id': None,
        'probability_next_same_behavior_state': None,
        'raw_probability': None,
        'root_probability': None,
        'smoothing_lambda': None,
        'support_user_count': None,
        'reason': None,
    }
    if not mappings:
        row['reason'] = 'no_last_behavior_mapping'
        return row
    if getattr(tree, 'collapse_consecutive_states', False):
        # Self-transitions were removed when the Tree was built, so P(B|B) is
        # structurally 0 and the smoothed value would only echo ROOT.
        row['reason'] = 'tree_built_with_collapse_consecutive_states'
        return row
    last = mappings[-1]
    if not bool(last.get('assignment_accepted')):
        row['reason'] = 'latest_behavior_state_mapping_rejected'
        return row
    sid = str(last['state_id'])
    row['last_state_id'] = sid
    root = tree.nodes[tree.root_id]
    next_node_id = root.get('children', {}).get(sid)
    if next_node_id is None or str(next_node_id) not in tree.nodes:
        row['reason'] = 'order1_context_not_observed_in_training'
        return row
    node = tree.nodes[str(next_node_id)]
    if int(node.get('context_order', -1)) != 1 or [str(x) for x in node.get('context_states', [])] != [sid]:
        raise ValueError('Corrupt Tree order-1 context: last-state node mismatch')
    p_raw = full_weight_probability(node, sid)
    p_root = full_weight_probability(root, sid)
    if p_raw is None or p_root is None:
        row['reason'] = 'full_collaborative_weights_unavailable'
        return row
    lam = float(node.get('smoothing_lambda', 1.0))
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f'Invalid Tree smoothing lambda={lam} for {sid}')
    p_stay = lam * p_raw + (1.0 - lam) * p_root
    if not -1e-9 <= p_stay <= 1+1e-9:
        raise ValueError(f'Invalid computed P(stay|{sid})={p_stay}')
    row.update({
        'available': True,
        'probability_next_same_behavior_state': float(min(1, max(0, p_stay))),
        'raw_probability': float(p_raw),
        'root_probability': float(p_root),
        'smoothing_lambda': float(lam),
        'support_user_count': int(node.get('support_user_count', 0)),
        'reason': 'order1_smoothed_full_collaborative_weights',
    })
    return row
