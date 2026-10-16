from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from bt4rec.common.io import fingerprint, read_jsonl
from bt4rec.common.text import clean_text
from bt4rec.common.vectors import l2_normalize
from bt4rec.data.items import as_str_list
from bt4rec.guidance.sources import how_only
from bt4rec.inference.constants import PMN_VERSION


def load_pmn_profiles(path: str, dataset: str, users: List[str], sequences: Dict[str, Any],
                      history_size: int, allowed_users: Optional[set], forbidden_users: set,
                      allow_debug: bool, train_how: Optional[Dict[str, List[Dict[str, Any]]]]=None):
    """Validate summary provenance without using any CoMemTree/Tree state in PMN matching."""
    expected = set(users)
    targets: Dict[str, str] = {}
    sources: List[Dict[str, Any]] = []
    seen = set()
    bad = Counter()
    for r in read_jsonl(path):
        if r.get('dataset') != dataset or r.get('summary_version') != PMN_VERSION:
            continue
        role, uid = r.get('role'), str(r.get('user_id'))
        if role not in ('target', 'source') or (role == 'target' and uid not in expected):
            continue
        if (role, uid) in seen:
            raise ValueError(f'Duplicate PMN profile {role}:{uid}')
        seen.add((role, uid))
        if not allow_debug and r.get('summary_backend') != 'gemma':
            raise ValueError(f'PMN {role}:{uid} is not a Gemma profile: {r.get("summary_backend")}')
        summary = clean_text(r.get('preference_summary'))
        if not summary:
            raise ValueError(f'PMN {role}:{uid}: blank summary')
        if uid not in sequences:
            raise ValueError(f'PMN {role}:{uid}: missing frozen sequence')
        tr = as_str_list(sequences[uid].get('train', []))
        hist = as_str_list(r.get('history_item_ids'))
        if not hist or int(r.get('history_size', -1)) != history_size:
            raise ValueError(f'PMN {role}:{uid}: wrong summary history length/config')
        if role == 'target':
            used = tr[-history_size:]
            key = fingerprint([PMN_VERSION, dataset, 'target', uid, history_size, used])
            if hist != used or r.get('summary_key') != key:
                raise ValueError(f'PMN target profile {uid} does not match current frozen observed train history')
            targets[uid] = summary
        else:
            # Source must be a training user, not a test-specific record.
            if allowed_users is not None and uid not in allowed_users:
                bad['outside_train_user_cohort'] += 1
                continue
            if uid in forbidden_users:
                bad['excluded_eval_source'] += 1
                continue
            next_item = str(r.get('terminal_train_item_id') or '')
            terminal_idx = r.get('terminal_window_index')
            if not next_item or terminal_idx is None:
                raise ValueError(f'PMN source {uid} missing continuation item/index metadata')
            raw_how = r.get('source_how')
            how = how_only(raw_how)
            matching_ends = [end for end in range(2,len(tr)+1)
                             if tr[end-1] == next_item and
                             tr[:end-1][-history_size:] == hist]
            if not matching_ends:
                raise ValueError(f'PMN source {uid}: summary prefix does not precede continuation in frozen train')
            key = fingerprint([PMN_VERSION, dataset, 'source', uid, history_size,
                                hist, int(terminal_idx), next_item, raw_how])
            if r.get('summary_key') != key:
                raise ValueError(f'PMN source {uid}: summary fingerprint mismatch')
            if train_how is not None:
                # Validate source continuation HOW against the actual train cache.
                # The terminal window index is an identifier, not a 0-based offset.
                available = train_how.get(uid, [])
                if not any(how == x['how'] and int(terminal_idx)==x['window_index'] for x in available):
                    raise ValueError(f'PMN source {uid}: HOW/index mismatch with training memory')
            sources.append({'uid':uid,'summary':summary,'how':how,
                            'terminal_train_item_id':next_item,
                            'terminal_window_index':int(terminal_idx),
                            'matched_prefix_end_positions':matching_ends})
    missing = sorted(expected - set(targets))
    if missing:
        raise ValueError(f'PMN summaries missing {len(missing)} targets; first={missing[:8]}')
    if len(sources) < 2:
        raise ValueError(f'PMN needs >=2 source users, found {len(sources)}, excluded={dict(bad)}')
    sources.sort(key=lambda x:x['uid'])
    return targets, sources, {'num_target_profiles':len(targets),'num_eligible_source_profiles':len(sources),
                               'excluded_source_profiles':dict(bad)}


def select_pmn_neighbours(encoder, targets: Dict[str, str], sources: List[Dict[str, Any]],
                          users: List[str]):
    # Embedding of textual preference summaries only. NO Tree, states or suffixes.
    texts = [targets[uid] for uid in users] + [s['summary'] for s in sources]
    mat = l2_normalize(encoder.encode(texts))
    q = mat[:len(users)]
    s = mat[len(users):]
    sims = q @ s.T
    neighbours = {}
    for i, uid in enumerate(users):
        row = sims[i].copy()
        for j, src in enumerate(sources):
            if src['uid'] == uid:
                row[j] = -np.inf
        winner = int(np.argmax(row))
        if not np.isfinite(row[winner]):
            raise ValueError(f'PMN {uid}: no eligible other training user')
        chosen = sources[winner]
        neighbours[uid] = {'neighbor_id':chosen['uid'], 'similarity':float(row[winner]),
                           'how':chosen['how'],
                           'continuation_train_item_id':chosen['terminal_train_item_id']}
    return neighbours
