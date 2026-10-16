from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bt4rec import shared
from bt4rec.common.io import fingerprint
from bt4rec.inference.constants import PMN_VERSION

# Legacy prepare_pmn_summaries.py wrote '..._gemma_v3_interaction_alignment' while the
# ablation loader only accepts PMN_VERSION ('..._gemma_v4_train_cache_metadata') and uses it
# inside summary_key, so every profile it produced was silently skipped. Use one constant.
SUMMARY_VERSION = PMN_VERSION


WINDOW_SIZE_DEFAULT = 3


def paths(dataset: str, data_root: str = 'data') -> dict[str, Path]:
    ds = shared._canonical_dataset(dataset)
    base = Path(data_root) / ds
    return {
        'items': base/'items.json', 'sequences': base/'user_sequences.json',
        'eval': base/'protocol'/'eval_users_1000.json',
        'candidates': base/'protocol'/'user_candidates_testpool1000_seed42.json',
    }


def canonical_how(x: dict[str, Any]) -> dict[str, str]:
    signature = shared.clean_text(x.get('trajectory_signature') or x.get('behavior_signature'))
    mech = shared.clean_text(x.get('mechanism'))
    scope = shared.clean_text(x.get('scope'))
    if not (signature and mech and scope):
        raise ValueError('Source memory missing trajectory_signature, mechanism or scope')
    return {
        'mechanism': shared.normalize_mechanism(mech),
        'scope': shared.normalize_scope(scope),
        'trajectory_signature': signature,
    }


def _simple_item_ids(value: Any) -> list[str] | None:
    """Only trust explicit item-ID arrays, never titles or generated text."""
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(v, (str, int)) for v in value):
        return None
    return [str(v) for v in value]


def _occurrences(sequence: list[str], subsequence: list[str]) -> list[int]:
    if not subsequence or len(subsequence) > len(sequence):
        return []
    n = len(subsequence)
    return [i for i in range(len(sequence) - n + 1)
            if sequence[i:i+n] == subsequence]


def _normalise_item_text(value: Any) -> str:
    """Simple exact-title matching after punctuation/whitespace normalization."""
    raw = unicodedata.normalize('NFKC', str(value or '')).casefold()
    return re.sub(r'[^\w]+', ' ', raw, flags=re.UNICODE).strip()


def _structured_sequence_ids(value: Any) -> list[str] | None:
    """Only return actual IDs, not title strings masquerading as IDs."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(('[', '{')):
            try:
                return _structured_sequence_ids(json.loads(stripped))
            except (ValueError, TypeError):
                return None
        return None
    if isinstance(value, dict):
        for key in ('item_ids', 'window_item_ids', 'sequence_item_ids', 'items',
                    'interactions', 'sequence', 'history'):
            if key in value:
                found = _structured_sequence_ids(value[key])
                if found:
                    return found
        return None
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(x, dict) for x in value):
        ids = []
        for x in value:
            v = next((x[k] for k in ('item_id', 'asin', 'product_id', 'movie_id', 'id')
                      if x.get(k) not in (None, '')), None)
            if v is None or isinstance(v, (list, dict)):
                return None
            ids.append(str(v))
        return ids
    # A naked list of strings could be TITLES, not IDs. Verify membership later.
    if all(isinstance(x, (str, int)) for x in value):
        return [str(x) for x in value]
    return None


def _sequence_titles(value: Any) -> list[str] | None:
    """Extract fully explicit per-item titles, never a generated behavior summary."""
    if isinstance(value, str):
        st = value.strip()
        if st.startswith(('[', '{')):
            try:
                return _sequence_titles(json.loads(st))
            except (ValueError, TypeError):
                pass
        return None
    if isinstance(value, dict):
        for key in ('items', 'interactions', 'sequence', 'history', 'item_titles', 'titles'):
            if key in value:
                found = _sequence_titles(value[key])
                if found:
                    return found
        return None
    if not isinstance(value, list) or not value:
        return None
    if all(isinstance(x, dict) for x in value):
        titles = [next((x[k] for k in ('title', 'name', 'item_title')
                        if isinstance(x.get(k), str) and x.get(k).strip()), None)
                  for x in value]
        return [str(x) for x in titles] if all(titles) else None
    if all(isinstance(x, str) for x in value):
        return list(value)
    return None


def _match_interaction_window(
    tr: list[str], items_meta: dict[str, Any], window_size: int,
    interaction_sequence: Any,
) -> tuple[int | None, str]:
    """Prove a unique train-window location from source interaction_sequence.

    Returns END POSITION (1-based exclusive) only if item IDs or all W exact
    normalized titles match ONE contiguous sequence within frozen TRAIN items.
    Never infer placement from a short local window_index alone.
    """
    if interaction_sequence is None:
        return None, 'missing_interaction_sequence'

    ids = _structured_sequence_ids(interaction_sequence)
    if ids and len(ids) == window_size and all(x in items_meta for x in ids):
        at = _occurrences(tr, ids)
        if len(at) == 1:
            return at[0] + window_size, 'interaction_sequence_item_ids'
        return None, 'ambiguous_interaction_sequence_ids'

    def title_for(iid: str) -> str:
        meta = items_meta.get(iid, {})
        if not isinstance(meta, dict):
            return ''
        return _normalise_item_text(meta.get('title') or meta.get('name'))

    titles = _sequence_titles(interaction_sequence)
    if titles and len(titles) == window_size:
        want = [_normalise_item_text(x) for x in titles]
        # Need reasonably descriptive strings to avoid generic-name collisions.
        if all(len(x) >= 4 for x in want):
            at = [start for start in range(len(tr) - window_size + 1)
                  if [title_for(i) for i in tr[start:start + window_size]] == want]
            if len(at) == 1:
                return at[0] + window_size, 'interaction_sequence_title_list'
            # Entries may include a label/category prefix or suffix for each
            # individual item. Require one full unique title per entry, in order.
            inclusive = [start for start in range(len(tr) - window_size + 1)
                         if all(re.search(r'(?<!\w)' + re.escape(title_for(iid)) + r'(?!\w)',
                                          wanted) is not None
                                for iid, wanted in zip(tr[start:start + window_size], want)
                                if len(title_for(iid)) >= 4)]
            # Short / missing titles cannot validate the entire window.
            inclusive = [start for start in inclusive
                         if all(len(title_for(i)) >= 4 for i in tr[start:start+window_size])]
            if len(inclusive) == 1:
                return inclusive[0] + window_size, 'interaction_sequence_labeled_title_list'
            return None, 'ambiguous_interaction_sequence_titles'

    # Unstructured precompute string, e.g. '1. Album A | ...; 2. Album B | ...'
    # Require EVERY full title, in chronological order, once per distinct
    # contiguous window. Partial keywords/genre matches are not accepted.
    if isinstance(interaction_sequence, str):
        text = interaction_sequence
    else:
        # For a list of titled dicts above, do not fall back to searching their
        # JSON representation, as it could contain metadata from other windows.
        return None, 'unmatched_structured_interaction_sequence'
    haystack = _normalise_item_text(text)
    possible = []
    for start in range(len(tr) - window_size + 1):
        candidate_titles = [title_for(i) for i in tr[start:start + window_size]]
        if any(len(t) < 4 for t in candidate_titles):
            continue
        cursor = 0
        good = True
        for t in candidate_titles:
            match = re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', haystack[cursor:])
            if match is None:
                good = False
                break
            cursor += match.end()
        if good:
            possible.append(start)
    if len(possible) == 1:
        return possible[0] + window_size, 'interaction_sequence_full_titles_in_order'
    return None, 'ambiguous_or_unmatched_interaction_sequence_text'


def training_memory_by_user(path: str) -> dict[str, list[dict]]:
    """Load precomputed dual-view memories, preserving provenance for alignment.

    One file may store one row per window OR one row per user.  The latter
    commonly contains the *actual precompute history* in train_history_item_ids,
    which may be a restricted suffix rather than the entire frozen train split.
    """
    by_user: dict[str, dict[int, dict]] = defaultdict(dict)
    for row in shared.read_jsonl(path):
        if row.get('precompute_ok') is False or row.get('parse_ok') is False:
            continue
        uid = row.get('user_id', row.get('uid'))
        if uid is None:
            continue
        uid = str(uid)
        if isinstance(row.get('generated_behaviors'), list):
            entries = row['generated_behaviors']
        else:
            b = next((row[k] for k in ('behavior','generated_behavior','local_memory','parsed_behavior',
                       'extracted_behavior','memory','output') if isinstance(row.get(k), dict)), row)
            entries = [b]
        cached_history = (_simple_item_ids(row.get('train_history_item_ids'))
                          or _simple_item_ids(row.get('history_item_ids')))
        for b in entries:
            if not isinstance(b, dict):
                continue
            try:
                how = canonical_how(b)
            except ValueError:
                continue
            raw_i = b.get('window_index', row.get('window_index'))
            if raw_i is None:
                continue
            try:
                ix = int(raw_i)
            except (TypeError, ValueError):
                continue
            full = dict(b)
            full['window_index'] = ix
            full['how'] = how
            full['_cached_train_history_item_ids'] = (_simple_item_ids(b.get('train_history_item_ids'))
                                                      or cached_history)
            full['_interaction_sequence'] = b.get('interaction_sequence', row.get('interaction_sequence'))
            full['_explicit_window_item_ids'] = (
                _simple_item_ids(b.get('window_item_ids'))
                or _simple_item_ids(row.get('window_item_ids'))
                or _simple_item_ids(b.get('item_ids'))
                or _simple_item_ids(row.get('item_ids')))
            # Deterministic, keep first successful record for each index.
            by_user[uid].setdefault(ix, full)
    return {u: [v for _, v in sorted(windows.items())] for u, windows in by_user.items() if windows}


def resolve_source_continuation(
    tr: list[str], windows: list[dict], window_size: int,
    window_index_offset: str | int = 'auto',
    items_meta: dict[str, Any] | None = None,
) -> tuple[dict | None, int | None, str, dict]:
    """Return (observed HOW, end_position_in_train, alignment_method, diagnostics).

    No source test/validation items are permitted. We only transfer an observed
    behavior whose window endpoint can be established from cache provenance or
    the full-history terminal index. The preference prefix ends BEFORE that
    window's final training item, never after it.
    """
    if not windows:
        return None, None, 'no_source_windows', {}
    n = len(tr)
    all_indices = sorted({int(b['window_index']) for b in windows})
    if window_index_offset == 'auto':
        # Common conventions: sliding-window start index (0/1 based) or
        # ending-item index (window_size-1 based). The first index is also
        # admissible if the cache contains a complete contiguous window grid.
        offsets = [0, 1, window_size-1]
        first = all_indices[0]
        if (len(all_indices) == n-window_size+1 and
                all_indices == list(range(first, first+len(all_indices)))):
            offsets = [first]
    else:
        offsets = [int(window_index_offset)]
        if offsets[0] < 0:
            raise ValueError('window_index_offset must be >= 0 or auto')
    diagnostics = {
        'train_length': n,
        'first_window_index': int(windows[0]['window_index']),
        'last_window_index': int(windows[-1]['window_index']),
        'full_terminal_0based_index': n-window_size,
        'full_terminal_1based_index': n-window_size+1,
        'has_cached_train_item_ids': any(bool(x.get('_cached_train_history_item_ids')) for x in windows),
        'has_explicit_window_item_ids': any(bool(x.get('_explicit_window_item_ids')) for x in windows),
        'has_interaction_sequence': any(x.get('_interaction_sequence') is not None for x in windows),
        'interaction_sequence_type': type(windows[-1].get('_interaction_sequence')).__name__,
    }
    if n <= window_size:
        return None, None, 'too_short', diagnostics

    # Try the latest cached windows first, requiring provable placement.
    # Crucially, old train_shared.jsonl caches often contain a short subset of
    # windows indexed 0..K: NOT the full source train history. The stored
    # interaction_sequence (when explicit enough) can recover actual placement.
    seq_failure_reasons = Counter()
    for b in reversed(windows):
        ix = int(b['window_index'])
        explicit = b.get('_explicit_window_item_ids')
        if explicit:
            if len(explicit) != window_size:
                continue
            positions = _occurrences(tr, explicit)
            if len(positions) == 1:
                end = positions[0] + window_size
                if end > 1:
                    return b, end, 'explicit_window_item_ids', diagnostics
            # Ambiguous repeated item windows cannot be placed safely.
            continue

        if items_meta is not None and b.get('_interaction_sequence') is not None:
            end, reason = _match_interaction_window(
                tr, items_meta, window_size, b.get('_interaction_sequence'))
            if end is not None and end > 1:
                return b, end, reason, diagnostics
            seq_failure_reasons[reason] += 1

        cached = b.get('_cached_train_history_item_ids')
        if cached:
            pos = _occurrences(tr, cached)
            if len(pos) != 1:
                continue
            base = pos[0]
            candidates: list[tuple[int, str]] = []
            for offset in offsets:
                local_start = ix-offset
                if 0 <= local_start and local_start+window_size <= len(cached):
                    candidates.append((base+local_start+window_size, f'cached_train_ids_offset{offset}'))
                global_start = ix-offset
                if 0 <= global_start and global_start+window_size <= n:
                    if base <= global_start and global_start+window_size <= base+len(cached):
                        candidates.append((global_start+window_size, f'global_index_verified_by_cache_offset{offset}'))
            if not candidates:
                continue
            # Resolve 0/1 indexing only when the cache establishes a terminal
            # window unambiguously. Earlier windows need an explicit offset.
            if window_index_offset == 'auto':
                # Complete cache over an explicitly supplied history has an
                # identifiable index offset even if indices start at 2, etc.
                if (len(all_indices) == len(cached)-window_size+1 and
                        all_indices == list(range(all_indices[0],all_indices[0]+len(all_indices)))):
                    candidates = [(end, method) for end,method in candidates
                                  if method.endswith(f'offset{all_indices[0]}')]
                terminal = [(end,method) for end,method in candidates if end == base+len(cached)]
                if terminal:
                    end,method = terminal[0]
                    if end > 1:
                        return b,end,method,diagnostics
                # If the stored source behavior is nonterminal, require the
                # precompute index base to be established from the min index.
                starts = {int(x['window_index']) for x in windows}
                inferred = min(starts) if min(starts) in offsets else None
                if inferred is None:
                    continue
                candidates = [(end,method) for end,method in candidates
                              if method.endswith(f'offset{inferred}')]
            unique_ends = {end for end,_ in candidates}
            if len(unique_ends) == 1:
                end,method = candidates[0]
                if end > 1:
                    return b,end,method,diagnostics
            continue

        # Without any item-ID provenance, an *earlier* window is not safe to
        # use: its index could refer to a truncated prefix or suffix. Only an
        # exact full-training-history terminal index is accepted.
        for offset in offsets:
            if ix == n-window_size+offset:
                # If index base can be observed, honor it.
                start_index = int(windows[0]['window_index'])
                if window_index_offset == 'auto' and start_index in offsets and offset != start_index:
                    continue
                return b,n,f'full_train_terminal_index_offset{offset}',diagnostics

    diagnostics['interaction_sequence_match_failures'] = dict(seq_failure_reasons)
    return None,None,'unverifiable_window_alignment',diagnostics


def make_cases(dataset: str, data_root: str, train_behaviors: str,
               history_size: int, window_size: int, window_index_offset: int,
               max_target_users: int, max_source_users: int,
               target_cache: str | None = None,
               train_user_ids: str | None = None) -> tuple[list[dict], dict]:
    p = paths(dataset, data_root)
    for key in ('items','sequences','eval','candidates'):
        if not p[key].is_file():
            raise FileNotFoundError(f'{key}: {p[key]}')
    if not Path(train_behaviors).is_file():
        raise FileNotFoundError(train_behaviors)
    items = shared.normalize_items(shared.load_json(p['items']))
    sequences = {str(u): v for u,v in shared.load_json(p['sequences']).items()}
    cohort = shared.load_user_ids_file(str(p['eval']))
    if not cohort or len(cohort) != len(set(cohort)):
        raise ValueError('Frozen eval cohort empty/duplicated')
    if max_target_users:
        if len(cohort) < max_target_users:
            raise ValueError(f'Frozen eval file has {len(cohort)} users, fewer than requested {max_target_users}')
        cohort = cohort[:max_target_users]
    candidates = shared.load_candidate_file(str(p['candidates']))
    cache = shared.load_precomputed_behaviors(target_cache) if target_cache else None
    source_memory = training_memory_by_user(train_behaviors)
    if not source_memory:
        raise ValueError(f'No valid source behaviors in {train_behaviors}')
    allowed_train = set(shared.load_user_ids_file(train_user_ids)) if train_user_ids else None
    cases: list[dict] = []
    for uid in cohort:
        if uid not in sequences:
            raise ValueError(f'Eval user {uid}: missing in sequences')
        seq = sequences[uid]
        train_ids = shared.as_str_list(seq.get('train'))
        target_ids = shared.as_str_list(seq.get('test'))
        candidate_ids, candidate_targets = shared.get_candidates_for_user(uid, seq, {}, candidates, 42)
        if len(target_ids) != 1 or candidate_targets != target_ids or len(candidate_ids) != 20:
            raise ValueError(f'{uid}: frozen test/candidate protocol mismatch')
        if len(train_ids) < 1:
            raise ValueError(f'{uid}: no train interactions')
        if cache is not None:
            if uid not in cache:
                raise ValueError(f'{uid}: missing behavior test cache')
            shared.validate_behavior_cache_history(uid, cache[uid], seq, 0, True)
        used = train_ids[-history_size:]
        key = fingerprint([SUMMARY_VERSION, shared._canonical_dataset(dataset), 'target', uid, history_size, used])
        cases.append({'role':'target', 'user_id':uid, 'history_ids': used, 'summary_key': key,
                      'source_how': None, 'terminal_window_index':None})
    excluded = Counter()
    aligned_by = Counter()
    source_count = 0
    problem_examples: list[dict] = []
    sources_nonfinal_in_train = 0
    for uid in sorted(source_memory):
        if allowed_train is not None and uid not in allowed_train:
            excluded['not_in_train_users_file'] += 1
            continue
        if uid not in sequences:
            excluded['no_sequence'] += 1
            continue
        tr = shared.as_str_list(sequences[uid].get('train'))
        terminal, end_pos, method, diag = resolve_source_continuation(
            tr, source_memory[uid], window_size, window_index_offset, items)
        if terminal is None or end_pos is None:
            excluded[method] += 1
            if len(problem_examples) < 6:
                problem_examples.append({'user_id':uid, 'reason':method, **diag,
                                         'record_fields':sorted(source_memory[uid][-1].keys())[:35],
                                         'interaction_sequence_example':str(source_memory[uid][-1].get('_interaction_sequence'))[:240]})
            continue
        # Source summary is causally prior to the continuation item at end_pos-1.
        # Even if the cache was computed over a limited subset, no later training
        # interactions are shown to the similarity retrieval model.
        used = tr[:end_pos-1][-history_size:]
        if not used:
            excluded['empty_source_prefix'] += 1
            continue
        how = terminal['how']
        terminal_idx = int(terminal['window_index'])
        terminal_train_item_id = tr[end_pos-1]
        key = fingerprint([SUMMARY_VERSION, shared._canonical_dataset(dataset), 'source', uid,
                           history_size, used, terminal_idx, terminal_train_item_id, how])
        cases.append({'role':'source','user_id':uid,'history_ids':used,'summary_key':key,
                      'source_how':how,'terminal_window_index':terminal_idx,
                      'terminal_train_item_id':terminal_train_item_id,
                      'source_alignment_method':method,'source_continuation_end_position':end_pos})
        aligned_by[method] += 1
        sources_nonfinal_in_train += int(end_pos < len(tr))
        source_count += 1
        if max_source_users and source_count >= max_source_users:
            break
    if source_count < 2:
        raise ValueError('Fewer than 2 source users with valid terminal training behaviors: '
                         f'{source_count}; exclusions={dict(excluded)}. '
                         f'Examples={problem_examples}. Check source JSONL provenance: this version will NOT arbitrarily select the last cached behavior.')
    # Match the main CoMemTree inference policy: missing item titles must NOT
    # invalidate the frozen user or change any item IDs / evaluation candidates.
    # PMN preference profiles are content summaries; avoid showing an ASIN as if
    # it were a meaningful title. Keep the actual category (if available), and
    # mark the unknown title explicitly instead of hallucinating metadata.
    missing_title_items: set[str] = set()
    missing_metadata_items: set[str] = set()
    missing_title_history_occurrences = Counter()
    title_fallback_examples: list[dict[str, str]] = []
    for case in cases:
        rows = []
        for iid in case['history_ids']:
            iid = str(iid)
            meta = items.get(iid)
            if not isinstance(meta, dict):
                missing_metadata_items.add(iid)
                # shared.get_item_info needs a dict; safely provide the same
                # unknown-metadata representation when metadata is unavailable.
                info = {'item_id': iid, 'title': 'Title unavailable',
                        'category': 'Unknown'}
            else:
                info = shared.get_item_info(iid, items)
                if not shared.clean_text(meta.get('title') or meta.get('name')):
                    info['title'] = 'Title unavailable'
            if info['title'] == 'Title unavailable':
                missing_title_items.add(iid)
                missing_title_history_occurrences[case['role']] += 1
                if len(title_fallback_examples) < 5:
                    title_fallback_examples.append({
                        'role': case['role'], 'item_id': iid,
                        'category': info['category']})
            rows.append(info)
        case['history_items'] = rows
    audit = {'dataset':shared._canonical_dataset(dataset), 'history_size':history_size,
             'window_size':window_size, 'window_index_offset':window_index_offset,
             'num_target':len(cohort),'num_source':source_count,
             'source_exclusions':dict(excluded), 'source_alignment_methods':dict(aligned_by),
             'sources_with_continuation_before_last_train_item':sources_nonfinal_in_train,
             'source_alignment_failure_examples':problem_examples,
             'missing_title_unique_item_count':len(missing_title_items),
             'missing_metadata_unique_item_count':len(missing_metadata_items),
             'missing_title_history_occurrences':dict(missing_title_history_occurrences),
             'missing_title_fallback':'Title unavailable | original category or Unknown',
             'missing_title_examples':title_fallback_examples,
             'overlap_source_target':len({c['user_id'] for c in cases if c['role']=='source'} & set(cohort)),
             'source_memory':str(Path(train_behaviors).resolve()),
             'source_pool_restricted_by_train_user_ids':bool(train_user_ids),
             'frozen_eval_file':str(p['eval']),'frozen_candidate_file':str(p['candidates']),
             'test_item_accessed_by_summary_model':False,
             'source_terminal_behavior_accessed_by_summary_model':False}
    return cases, audit


def summary_prompt(history_items: list[dict]) -> str:
    return f'''Summarize the user's STABLE AND RECENT CONTENT PREFERENCES from the
observed items below. This is a profile for SIMILAR-USER retrieval, not a
prediction of the user's next action.

Observed chronological history (oldest to newest):
{shared.format_history(history_items)}

Return ONLY JSON: {{"preference_summary":"..."}}
Write one concise English summary (30-75 words), emphasizing recurring categories,
genres, topics, styles, and stable preferences plus notable recent interests.
Ground the summary in the listed items; do not invent user attributes or motivations.
Avoid listing item titles, IDs, dates, or exact products. Do NOT predict next items
or future behavior. Do not include any test/candidate information.'''


def decode_summary(raw: str) -> str:
    x = shared.extract_json_value(raw)
    if not isinstance(x, dict) or not isinstance(x.get('preference_summary'), str):
        raise ValueError('LLM must output JSON with preference_summary')
    val = shared.clean_text(x['preference_summary'])
    if not val or len(val) > 1500:
        raise ValueError('Empty or excessively long preference_summary')
    return val
