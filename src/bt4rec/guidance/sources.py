from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from bt4rec.common.io import fingerprint, read_jsonl
from bt4rec.common.text import clean_text
from bt4rec.data.items import as_str_list
from bt4rec.taxonomy import normalize_mechanism, normalize_scope


def how_only(raw: Dict[str, Any]) -> Dict[str, str]:
    """Uniform mechanism/scope/trajectory; no user IDs, IDs, content or probabilities."""
    raw = raw or {}
    s = clean_text(raw.get('trajectory_signature') or raw.get('behavior_signature')
                   or raw.get('canonical_behavior_signature') or raw.get('representative_signature'))
    m = clean_text(raw.get('mechanism'))
    scope = clean_text(raw.get('scope'))
    if not all([s, m, scope]):
        raise ValueError(f'HOW missing mechanism/scope/signature: {raw}')
    return {'mechanism': normalize_mechanism(m), 'scope': normalize_scope(scope),
            'trajectory_signature': s}


def read_train_how(path: str, allowed_users: Optional[set],
                   forbidden_users: set) -> Dict[str, List[Dict[str, Any]]]:
    grouped = defaultdict(dict)
    for row in read_jsonl(path):
        uid_raw = row.get('user_id', row.get('uid'))
        if uid_raw is None:
            continue
        uid = str(uid_raw)
        if (allowed_users is not None and uid not in allowed_users) or uid in forbidden_users:
            continue
        if row.get('precompute_ok') is False:
            continue
        if isinstance(row.get('generated_behaviors'), list):
            windows = row['generated_behaviors']
        else:
            b = next((row[k] for k in ('behavior','generated_behavior','local_memory',
                          'parsed_behavior','extracted_behavior','memory','output')
                      if isinstance(row.get(k), dict)), row)
            windows = [b]
        for idx, b in enumerate(windows):
            if not isinstance(b, dict):
                continue
            try:
                how = how_only(b)
            except ValueError:
                continue
            wi = b.get('window_index', row.get('window_index', idx))
            try:
                wi = int(wi)
            except (ValueError, TypeError):
                continue
            grouped[uid].setdefault(wi, how)
    result = {uid: [{'window_index':wi,'how':b} for wi,b in sorted(w.items())] for uid,w in grouped.items() if w}
    if not result:
        raise ValueError(f'No eligible train-only source HOW in {path}. Check cohort/exclusions.')
    return result


def choose_random_how(training: Dict[str, List[Dict[str, Any]]], uid: str, seed: int):
    # Uniform sampling of SOURCE USER then one observed behavior window;
    # deterministic per target user and seed, independent of target/test labels.
    available = sorted(x for x in training if x != uid)
    if not available:
        raise ValueError(f'{uid}: random source has no other user')
    src = available[int(fingerprint([seed, uid, 'random_user'])[:16], 16) % len(available)]
    seq = training[src]
    ix = int(fingerprint([seed, uid, 'random_window'])[:16], 16) % len(seq)
    return seq[ix]['how'], {'source_user_id': src, 'source_window_index': seq[ix]['window_index'],
                            'selection': 'uniform_train_user_window'}


def tree_global_how(tree):
    node = tree.nodes[tree.root_id]
    dist = node.get('smoothed_next_distribution') or node.get('raw_next_distribution') or []
    if not dist:
        raise ValueError('Tree ROOT next-state distribution empty')
    return how_only(tree.vocab.state_evidence(str(dist[0]['state_id']))), {
        'global_state_id':str(dist[0]['state_id']),
        'global_probability':float(dist[0].get('probability',0.0)),
        'root_support_user_count':int(node.get('support_user_count',0)),
    }


def load_oracle(path: str, users: List[str], sequences: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    expected = set(users)
    out = {}
    for row in read_jsonl(path):
        uid = str(row.get('user_id'))
        if uid not in expected:
            continue
        if uid in out:
            raise ValueError(f'Duplicate future oracle for {uid}')
        if row.get('future_aware') is not True:
            raise ValueError(f'{uid}: oracle record must declare future_aware=true')
        target = as_str_list(sequences[uid].get('test', []))
        if len(target) != 1 or str(row.get('target_item_id')) != target[0]:
            raise ValueError(f'{uid}: oracle target ID does not match frozen test item')
        out[uid] = how_only(row.get('oracle_behavior', {}))
    missing = sorted(expected - set(out))
    if missing:
        raise ValueError(f'Oracle cache missing {len(missing)} target users, first={missing[:10]}')
    return out
