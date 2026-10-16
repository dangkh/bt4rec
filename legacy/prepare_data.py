#!/usr/bin/env python3
"""Create a metadata-safe, NO-VALIDATION CoMemTree frozen protocol.

No changes to items.json, user_sequences.json, or user_negatives.json.
Train and eval lists contain identical user IDs; ranking observes TRAIN events
only, while the unique TEST item is held out as the target. Oracle-HOW is
extracted from [train[-2], train[-1], test[0]] and never the obsolete val field.
Candidate sets comprise 1 held-out test item + K-1 eligible frozen negatives.

Strict title validity: items[id] must be a dict with non-blank string 'title'.
No 'name' or 'Item <id>' fallback is counted as valid.

Examples
--------
python prepare_frozen_protocol.py --dataset CDs \
    --sizes 100,300,500,1000 --seed 42

# When too few per-user negatives have valid metadata:
# python prepare_frozen_protocol.py --dataset CDs --negative-fallback pooled

# Output goes to data/CDs/protocol/ ONLY if >=1000 eligible users.
# Never silently overwrite old protocol files.
python prepare_frozen_protocol.py --dataset CDs \
    --sizes 100,300,500,1000 --negative-fallback pooled --seed 42
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from itertools import accumulate
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def item_ids(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for value_item in value:
            out.extend(item_ids(value_item, field))
        return out
    if isinstance(value, dict):
        for key in ('item_ids', 'items', 'item_id', 'item'):
            if key in value:
                return item_ids(value[key], field + '.' + key)
        raise ValueError(f'{field}: cannot interpret item ID dict {list(value)[:8]}')
    raise ValueError(f'{field}: unsupported item-ID type {type(value).__name__}')


def read_user_ids(path: Path) -> list[str]:
    content = path.read_text(encoding='utf-8').strip()
    if not content:
        return []
    try:
        obj = json.loads(content)
    except json.JSONDecodeError:
        return list(dict.fromkeys(line.strip() for line in content.splitlines() if line.strip()))
    if isinstance(obj, list):
        return list(dict.fromkeys(str(x) for x in obj))
    if isinstance(obj, dict):
        if isinstance(obj.get('users'), list):
            return list(dict.fromkeys(str(x) for x in obj['users']))
        return list(dict.fromkeys(str(x) for x in obj.keys()))
    raise ValueError(f'User file must be list/dict/lines: {path}')


def valid_title(iid: str, items: dict[str, Any]) -> bool:
    meta = items.get(iid)
    return isinstance(meta, dict) and isinstance(meta.get('title'), str) and bool(meta['title'].strip())


def valid_category(iid: str, items: dict[str, Any]) -> bool:
    meta = items.get(iid)
    if not isinstance(meta, dict):
        return False
    val = meta.get('main_cat') or meta.get('category') or meta.get('categories')
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, tuple)):
        return bool(val) and any(str(v).strip() for v in val if v is not None)
    return False


def get_val(seq: dict[str, Any], uid: str) -> list[str]:
    values = [item_ids(seq[k], f'{uid}.{k}') for k in ('val', 'valid', 'validation', 'validation_item_ids', 'val_item_ids') if k in seq]
    nonempty = [x for x in values if x]
    if len(nonempty) > 1 and any(x != nonempty[0] for x in nonempty[1:]):
        raise ValueError(f'{uid}: contradictory validation fields')
    return nonempty[0] if nonempty else []


def negative_ids(entry: Any, uid: str) -> list[str]:
    # user_negatives.json usually has {user_id: {"test_neg": [item_ids...]}}
    if isinstance(entry, dict):
        for key in ('test_neg', 'test_negatives', 'negatives', 'negative_items'):
            if key in entry:
                return item_ids(entry[key], f'negative:{uid}.{key}')
        raise ValueError(f'user_negatives[{uid}] has no test_neg/negatives: keys {list(entry)[:8]}')
    return item_ids(entry, f'negative:{uid}')


def seed_for_user(seed: int, uid: str) -> int:
    digest = hashlib.sha256(f'{seed}|{uid}|title-safe-v1'.encode()).digest()
    return int.from_bytes(digest[:8], 'big')


def parse_sizes(text: str) -> list[int]:
    sizes = sorted(set(int(s.strip()) for s in text.split(',') if s.strip()))
    if not sizes or any(x < 1 for x in sizes):
        raise ValueError('--sizes should be comma separated positive integers')
    return sizes


def save_json(path: Path, payload: Any, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite existing file: {path}; pass --overwrite to allow')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write('\n')


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Generate frozen user lists and metadata-safe candidate pools')
    ap.add_argument('--dataset', help='E.g. CDs, Video_Games, Books')
    ap.add_argument('--data-dir', type=Path, help='Default: data/<dataset>')
    ap.add_argument('--items', type=Path, help='Default <data-dir>/items.json')
    ap.add_argument('--sequences', type=Path, help='Default <data-dir>/user_sequences.json')
    ap.add_argument('--negatives', type=Path, help='Default <data-dir>/user_negatives.json')
    ap.add_argument('--sizes', default='100,300,500,1000')
    ap.add_argument('--candidate-size', type=int, default=20,
                    help='Exactly this many candidates per user, including test item')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--min-train-length', type=int, default=3,
                    help='At least two train items required for Oracle HOW; default 3 for CoMemTree windows')
    ap.add_argument('--priority-users-file', type=Path,
                    help='Prioritize these user IDs to retain existing cohort; default existing eval_users_1000.json if present; pass --no-priority-users to disable')
    ap.add_argument('--no-priority-users', action='store_true')
    ap.add_argument('--reuse-candidates-file', type=Path,
                    help='Reuse unchanged existing candidate orders for eligible users whenever all candidates have titles; default existing <data-dir>/user_candidates_testpool1000_seed<seed>.json if present')
    ap.add_argument('--no-reuse-existing-candidates', action='store_true', help='Rebuild ALL candidates from user_negatives rather than reusing valid frozen rows')
    ap.add_argument('--output-dir', type=Path,
                    help='Default <data-dir>/protocol (never silently overwrite originals)')
    ap.add_argument('--overwrite', action='store_true', help='Allow replacing output files')
    ap.add_argument('--no-require-window-titles', dest='require_window_titles', action='store_false',
                    help='Only require test+candidate titles; by default require BOTH last two TRAIN titles for Oracle HOW')
    ap.set_defaults(require_window_titles=True)
    ap.add_argument('--require-all-history-titles', action='store_true',
                    help='Additionally require ALL train items to have valid title (more restrictive)')
    ap.add_argument('--require-category', action='store_true', help='Require a nonblank category for all candidates + Oracle window as well')
    ap.add_argument('--negative-fallback', choices=['none', 'pooled', 'catalog'], default='none',
                    help='none = only per-user negatives; pooled = backfill from globally pooled user_negatives (recommended); catalog = sample from all titled catalog items. Both fallback modes change sampling distribution.')
    ap.add_argument('--dry-run', action='store_true', help='Check eligibility and print counts without writing outputs')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset and not args.data_dir:
        raise ValueError('Provide --dataset or --data-dir')
    if args.candidate_size < 2:
        raise ValueError('--candidate-size must be >= 2')
    if args.min_train_length < 1:
        raise ValueError('--min-train-length must be >= 1')
    sizes = parse_sizes(args.sizes)
    dataset_dir = args.data_dir or Path('data') / args.dataset
    outdir = args.output_dir or (dataset_dir / 'protocol')
    ipath = args.items or dataset_dir / 'items.json'
    spath = args.sequences or dataset_dir / 'user_sequences.json'
    npath = args.negatives or dataset_dir / 'user_negatives.json'

    items_raw = read_json(ipath)
    seq_raw = read_json(spath)
    neg_raw = read_json(npath)
    if not all(isinstance(x, dict) for x in (items_raw, seq_raw, neg_raw)):
        raise ValueError('items.json, user_sequences.json and user_negatives.json must be maps')
    items = {str(k): v for k, v in items_raw.items()}
    sequences = {str(k): v for k, v in seq_raw.items()}
    negatives = {str(k): v for k, v in neg_raw.items()}
    titled_catalog = sorted(iid for iid in items if valid_title(iid, items) and (not args.require_category or valid_category(iid, items)))

    def valid_metadata(iid: str) -> bool:
        return valid_title(iid, items) and (not args.require_category or valid_category(iid, items))
    rng = random.Random(args.seed)

    # When per-user negatives are too sparse after metadata filtering, backfill
    # from items that actually occur in the ORIGINAL user_negatives.json.
    # Weight by how many users have the negative in their original pool, which
    # approximately preserves the original marginal negative-source distribution.
    # Candidate selections remain deterministic at fixed seed and user ID.
    pooled_ids: list[str] = []
    pooled_cumulative: list[int] = []
    pooled_total_weight = 0
    if args.negative_fallback == 'pooled':
        global_counts: Counter[str] = Counter()
        malformed_pool_rows = 0
        for pool_uid, neg_entry in negatives.items():
            try:
                ids = negative_ids(neg_entry, pool_uid)
            except (ValueError, TypeError):
                malformed_pool_rows += 1
                continue
            global_counts.update(set(iid for iid in ids if valid_metadata(iid)))
        if not global_counts:
            raise ValueError('No valid metadata-covered negatives found in user_negatives.json; use --negative-fallback catalog or repair metadata.')
        pooled_ids = sorted(global_counts)
        pooled_cumulative = list(accumulate(global_counts[iid] for iid in pooled_ids))
        pooled_total_weight = pooled_cumulative[-1]
        print(f'[NEG-POOL] {len(pooled_ids):,} distinct metadata-valid global negative items; '
              f'{pooled_total_weight:,} total user-level negative appearances; '
              f'{malformed_pool_rows:,} malformed records ignored.')

    priority_path = args.priority_users_file
    if priority_path is None and not args.no_priority_users:
        # The new protocol is in <data-dir>/protocol; retain as many already
        # evaluated users as possible when regenerating the cohort.
        for default_path in (outdir / f'eval_users_{max(sizes)}.json',
                             dataset_dir / f'eval_users_{max(sizes)}.json'):
            if default_path.is_file():
                priority_path = default_path
                break
    priority = read_user_ids(priority_path) if priority_path else []
    old_candidate_path = args.reuse_candidates_file
    if old_candidate_path is None and not args.no_reuse_existing_candidates:
        for existing_candidates in (
            outdir / f'user_candidates_testpool1000_seed{args.seed}.json',
            dataset_dir / f'user_candidates_testpool1000_seed{args.seed}.json',
        ):
            if existing_candidates.is_file():
                old_candidate_path = existing_candidates
                break
    old_candidates: dict[str, Any] = {}
    if old_candidate_path is not None:
        candidate_data = read_json(old_candidate_path)
        if not isinstance(candidate_data, dict):
            raise ValueError(f'{old_candidate_path} must be a dict keyed by user ID')
        old_candidates = {str(k): v for k, v in candidate_data.items()}
    # Keep old cohort when possible; sample backfills with deterministic seed.
    # Sorting before shuffle means reproducible order independent of JSON insertion.
    priority = list(dict.fromkeys(uid for uid in priority if uid in sequences))
    priority_set = set(priority)
    rest = sorted(set(sequences) - priority_set)
    rng.shuffle(rest)
    ordered_uids = priority + rest

    exclusions: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}
    viable: list[str] = []
    selected_candidates: dict[str, dict[str, Any]] = {}
    fallback_users: set[str] = set()
    fallback_added: dict[str, int] = {}
    reused_users: set[str] = set()
    selected_missing_history: dict[str, int] = {}

    def reject(reason: str, uid: str) -> None:
        exclusions[reason] += 1
        if len(examples.setdefault(reason, [])) < 6:
            examples[reason].append(uid)

    for uid in ordered_uids:
        entry = sequences[uid]
        if not isinstance(entry, dict):
            reject('invalid_sequence_record', uid)
            continue
        try:
            train = item_ids(entry.get('train'), f'{uid}.train')
            # An obsolete `val` field may still occur in raw sequences; it is
            # NEVER used as observed history or as part of the Oracle window.
            # But exclude such known positives from sampled negatives for safety.
            unused_legacy_val = get_val(entry, uid)
            test = item_ids(entry.get('test'), f'{uid}.test')
        except (ValueError, TypeError):
            reject('malformed_sequence', uid)
            continue
        if len(train) < args.min_train_length:
            reject('insufficient_train_events', uid)
            continue
        if len(test) != 1:
            reject('test_not_single_event', uid)
            continue
        target = test[0]
        if not valid_title(target, items):
            reject('test_missing_title', uid)
            continue
        if args.require_category and not valid_category(target, items):
            reject('test_missing_category', uid)
            continue
        # EXACTLY the window used by oracle_probe_v2.py:
        # [train[-2], train[-1], test[0]].  Do not screen validation items.
        if args.require_window_titles and (len(train) < 2 or not all(valid_metadata(iid) for iid in train[-2:])):
            reject('last_two_train_missing_title', uid)
            continue
        missing_history = sum(not valid_metadata(iid) for iid in train)
        if args.require_all_history_titles and missing_history:
            reject('history_missing_title', uid)
            continue
        # Legacy val (if present) is excluded from *negative* sampling only.
        # It is NOT an observed history item and is NOT an Oracle item.
        forbidden = set(train + unused_legacy_val + test)
        existing_row = old_candidates.get(uid)
        if isinstance(existing_row, dict):
            try:
                existing_ids = item_ids(existing_row.get('candidates', existing_row.get('candidate_item_ids', [])), f'{uid}.existing_candidates')
                old_target = item_ids(existing_row.get('target', existing_row.get('ground_truth', [])), f'{uid}.existing_target')
                if (len(existing_ids) == args.candidate_size
                    and len(set(existing_ids)) == args.candidate_size
                    and existing_ids.count(target) == 1
                    and (not old_target or old_target == [target])
                    and not (set(existing_ids) - {target}) & forbidden
                    and all(valid_metadata(iid) for iid in existing_ids)):
                    viable.append(uid)
                    selected_candidates[uid] = {'target': [target], 'candidates': existing_ids}
                    selected_missing_history[uid] = missing_history
                    reused_users.add(uid)
                    continue
            except (ValueError, TypeError):
                pass
        if uid not in negatives and args.negative_fallback == 'none':
            reject('user_negatives_missing', uid)
            continue
        try:
            raw_neg = negative_ids(negatives[uid], uid) if uid in negatives else []
        except (ValueError, TypeError):
            if args.negative_fallback == 'none':
                reject('malformed_user_negatives', uid)
                continue
            raw_neg = []  # Metadata-safe fallback handles unavailable raw negatives.

        # Do NOT count unobserved user negatives as true dislikes: they are candidate decoys.
        valid_negs = list(dict.fromkeys(
            x for x in raw_neg if x not in forbidden and valid_metadata(x)
        ))
        needed = args.candidate_size - 1
        if len(valid_negs) < needed and args.negative_fallback in ('pooled', 'catalog'):
            # Keep every usable original negative in the eligible sampling pool;
            # supplement only the shortage. NEVER sample known user positives.
            original_valid_n = len(valid_negs)
            used = forbidden.union(valid_negs)
            fallback_rng = random.Random(seed_for_user(args.seed + 93131, uid))
            tries = 0
            max_tries = max(3000, needed * 200)
            while len(valid_negs) < needed and tries < max_tries:
                tries += 1
                if args.negative_fallback == 'pooled':
                    index = bisect.bisect_right(
                        pooled_cumulative,
                        fallback_rng.randrange(pooled_total_weight)
                    )
                    iid = pooled_ids[index]
                else:
                    if not titled_catalog:
                        break
                    iid = titled_catalog[fallback_rng.randrange(len(titled_catalog))]
                if iid not in used:
                    valid_negs.append(iid)
                    used.add(iid)
            if len(valid_negs) >= needed:
                fallback_users.add(uid)
                fallback_added[uid] = len(valid_negs) - original_valid_n
        if len(valid_negs) < needed:
            reject('insufficient_titled_user_negatives', uid)
            continue

        per_user_rng = random.Random(seed_for_user(args.seed, uid))
        sampled = per_user_rng.sample(valid_negs, needed)
        candidates = sampled + [target]
        per_user_rng.shuffle(candidates)
        # Fail closed on any programming mistakes or unexpected input values.
        if (len(candidates) != args.candidate_size or
            len(set(candidates)) != len(candidates) or
            candidates.count(target) != 1 or
            any(not valid_metadata(iid) for iid in candidates) or
            any(iid in forbidden for iid in sampled)):
            raise AssertionError(f'Candidate pool check failed for uid={uid}')

        viable.append(uid)
        selected_candidates[uid] = {'target': [target], 'candidates': candidates}
        selected_missing_history[uid] = missing_history

    size_max = sizes[-1]
    if len(viable) < size_max:
        print(f'[ERROR] Only {len(viable):,} eligible users for requested {size_max:,}.', file=sys.stderr)
        print(f'Exclusion reasons: {dict(exclusions)}', file=sys.stderr)
        print('Cannot guarantee requested sample size with the current metadata and negatives.', file=sys.stderr)
        print('Try --negative-fallback pooled (preferred), --negative-fallback catalog, restoring metadata or lowering --sizes.', file=sys.stderr)
        return 2

    selected = viable[:size_max]
    cand_out = {uid: selected_candidates[uid] for uid in selected}
    prioritized_kept = sum(uid in priority_set for uid in selected)
    # Independent full verification of ALL generated cohort sizes.
    # This duplicates Oracle's title guard and ensures future code edits cannot
    # silently allow a title-less train[-2] into the frozen cohort.
    for size in sizes:
        user_subset = selected[:size]
        assert len(user_subset) == size and len(set(user_subset)) == size
        for uid in user_subset:
            row = cand_out[uid]
            tr = item_ids(sequences[uid].get('train'), f'{uid}.train')
            te = item_ids(sequences[uid].get('test'), f'{uid}.test')
            assert len(tr) >= 2 and len(te) == 1
            if args.require_window_titles:
                assert all(valid_metadata(iid) for iid in [tr[-2], tr[-1], te[0]])
            assert row['target'] == te
            assert len(row['target']) == 1 and row['target'][0] in row['candidates']
            assert len(row['candidates']) == args.candidate_size
            assert all(valid_metadata(iid) for iid in row['candidates'])

    report = {
        'dataset': args.dataset or dataset_dir.name,
        'seed': args.seed,
        'candidate_size': args.candidate_size,
        'sizes': sizes,
        'train_eval_users_same': True,
        'protocol_note': 'Train/eval user lists are identical; observed history is TRAIN only, held-out target is TEST. Any legacy val field is ignored for inference and Oracle (excluded from negatives only).',
        'oracle_window_policy': 'train[-2], train[-1], test[0]',
        'uses_validation_for_oracle_or_ranking': False,
        'negative_source': str(npath),
        'negative_fallback': args.negative_fallback,
        'users_with_fallback_in_selected': sum(uid in fallback_users for uid in selected),
        'fallback_negatives_added_in_selected': sum(fallback_added.get(uid, 0) for uid in selected),
        'global_negative_pool_unique_items': len(pooled_ids) if args.negative_fallback == 'pooled' else None,
        'global_negative_pool_user_appearances': pooled_total_weight if args.negative_fallback == 'pooled' else None,
        'require_window_titles': args.require_window_titles,
        'require_category': args.require_category,
        'require_all_history_titles': args.require_all_history_titles,
        'input': {'items': str(ipath), 'sequences': str(spath), 'negatives': str(npath),
                  'priority_user_list': str(priority_path) if priority_path else None},
        'total_users_in_sequences': len(sequences),
        'eligible_user_count': len(viable),
        'exclusions': dict(sorted(exclusions.items())),
        'exclusion_examples': examples,
        'old_users_in_new_cohort': prioritized_kept,
        'new_users_in_new_cohort': size_max - prioritized_kept,
        'reused_candidate_rows': sum(uid in reused_users for uid in selected),
        'newly_sampled_candidate_rows': sum(uid not in reused_users for uid in selected),
        'reuse_candidates_from': str(old_candidate_path) if old_candidate_path else None,
        'selected_users_with_missing_history_title_or_category': sum(
            selected_missing_history[uid] > 0 for uid in selected
        ),
        'selected_total_history_items_missing_title_or_category': sum(
            selected_missing_history[uid] for uid in selected
        ),
        'selected_candidate_titles_verified': True,
        'selected_test_titles_verified': True,
        'inference_candidate_schema': {'target': ['test_item_id'], 'candidates': ['shuffled_20_item_ids']},
        'warning': ('These are new protocol outputs. Do NOT mix with older precomputed behaviors, '
                    'Tree checkpoints, baseline results, or previous frozen candidate files until revalidated.'),
    }
    # Preflight: verify all destination paths before writing anything.
    paths: list[Path] = []
    for size in sizes:
        paths += [outdir / f'train_users_{size}.json', outdir / f'eval_users_{size}.json']
    paths += [outdir / f'user_candidates_testpool1000_seed{args.seed}.json',
              outdir / 'frozen_protocol_summary.json',
              outdir / 'excluded_user_examples.csv']
    existing = [str(p) for p in paths if p.exists()]
    if existing and not args.overwrite and not args.dry_run:
        print('[ERROR] Some output files already exist; aborting before writing:', file=sys.stderr)
        for p in existing[:8]:
            print('   ', p, file=sys.stderr)
        print('Use --overwrite only if you intend to replace them.', file=sys.stderr)
        return 2

    print(f'Dataset: {dataset_dir} | all sequence users: {len(sequences):,}')
    print(f'Eligible users: {len(viable):,} | selected: {size_max:,} | title-safe candidates: {args.candidate_size}/user')
    print(f'Old cohort retained: {prioritized_kept:,}/{size_max:,}; replacement users: {size_max-prioritized_kept:,}')
    print(f'Existing valid candidate rows reused: {report["reused_candidate_rows"]:,}/{size_max:,}')
    print(f'Users needing negative fallback: {report["users_with_fallback_in_selected"]:,}; '
          f'negative items supplemented: {report["fallback_negatives_added_in_selected"]:,}')
    print(f'Rejected: {dict(exclusions)}')
    print('Train/eval user IDs: SAME; Oracle window=train[-2],train[-1],test[0]; no validation.')
    print(f'Output directory: {outdir}')
    if args.dry_run:
        print('[DRY RUN] No files written.')
        return 0

    for size in sizes:
        user_subset = selected[:size]
        save_json(outdir / f'train_users_{size}.json', user_subset, args.overwrite)
        save_json(outdir / f'eval_users_{size}.json', user_subset, args.overwrite)
    save_json(outdir / f'user_candidates_testpool1000_seed{args.seed}.json', cand_out, args.overwrite)
    save_json(outdir / 'frozen_protocol_summary.json', report, args.overwrite)
    with (outdir / 'excluded_user_examples.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['reason', 'user_id'])
        for reason, ids in sorted(examples.items()):
            for uid in ids:
                writer.writerow([reason, uid])
    print('[OK] Wrote nested cohorts, frozen candidates and audit report.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, FileNotFoundError, FileExistsError, json.JSONDecodeError) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr)
        sys.exit(2)
