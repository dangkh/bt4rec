"""Unified CoMemTree behaviour precompute (train-only windows, no validation).

Replaces precompute_behavior.py (``--backend gemma``) and
precompute_train_test_shared_qwen.py (``--backend qwen``). Both backends share
windowing, caching and record packing; only the prompt, the LLM loader and the
validation/retry policy differ (see bt4rec/behavior/gemma.py and qwen.py).

Outputs (same schema as before):
    precomputed/<dataset>[/qwen]/train_shared.jsonl   window-level memory for the Tree
    precomputed/<dataset>[/qwen]/test_shared.jsonl    user-level memory for inference

Example:
    python -m bt4rec.cli.precompute --backend qwen --dataset CDs \
        --user-ids-file data/CDs/protocol/eval_users_1000.json --history-size 20 --plan-only
"""
import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm

from bt4rec.behavior import gemma, qwen
from bt4rec.behavior.records import to_test_row, to_train_record
from bt4rec.behavior.windows import (
    Window,
    as_ids,
    get_info,
    make_train_windows,
    read_ids,
    validate_selected_users,
    window_key,
)
from bt4rec.common.io import load_json, write_jsonl_atomic

BACKENDS = {"gemma": gemma, "qwen": qwen}


def parse_args(argv=None) -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--backend", choices=sorted(BACKENDS), default="gemma")
    known, _ = pre.parse_known_args(argv)
    be = BACKENDS[known.backend]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=sorted(BACKENDS), default="gemma",
                   help="gemma = legacy precompute_behavior.py; qwen = precompute_train_test_shared_qwen.py")
    p.add_argument("--dataset", help="Dataset name, e.g. CDs, Video_Games, Books, ML")
    p.add_argument("--data-root", default="data", help="Root folder containing dataset subfolders")
    p.add_argument("--data-dir", help="Optional direct dataset folder; alternative to --dataset")
    p.add_argument("--sequences", help="Optional input override; defaults to <data-dir>/user_sequences.json")
    p.add_argument("--items", help="Optional input override; defaults to <data-dir>/items.json")
    p.add_argument("--sequences-file", default="user_sequences.json", help="Filename inside data-dir")
    p.add_argument("--items-file", default="items.json", help="Filename inside data-dir")
    p.add_argument("--user-ids-file", help="Common protocol user list for both outputs; auto-uses protocol/eval_users_1000.json if available")
    p.add_argument("--number-of-users", type=int, default=0, help="Select first N users for BOTH outputs; 0 = all")
    p.add_argument("--output-root", default="precomputed", help="Root for automatically named output files")
    p.add_argument("--output-dir", help="Optional output folder override")
    p.add_argument("--existing-train", help="Old training memories (optional; auto-detect known filenames if omitted)")
    p.add_argument("--no-auto-existing-train", action="store_true", help="Disable auto-discovery of old training memories")
    p.add_argument("--cache-output", help="Durable incremental cache; auto-detected on restart")
    p.add_argument("--train-output", help="Optional output override (default: <output-dir>/train_shared.jsonl)")
    p.add_argument("--test-output", help="Optional output override (default: <output-dir>/test_shared.jsonl)")
    p.add_argument("--history-size", type=int, default=20, help="0 = unlimited; cap raw items BEFORE metadata filtering")
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--behavior-domain", choices=["generic", "books"], default="generic",
                   help="gemma backend only: domain-specific semantic examples in the extraction "
                        "prompt (books = genre/series/author/topic). Changes the prompt version, so "
                        "use a NEW --output-dir; Tree and inference must use the same variant.")
    p.add_argument("--recent-behaviors", type=int, default=5)
    p.add_argument("--plan-only", action="store_true", help="Audit input and cache reuse; no LLM/GPU needed")
    p.add_argument("--max-attempts", type=int, default=be.DEFAULTS["max_attempts"],
                   help="Qwen only: retry malformed/invalid extractions")
    p.add_argument("--overwrite", action="store_true", help="Atomically replace existing finished train/test outputs")
    p.add_argument("--model-name", default=be.DEFAULTS["model_name"])
    p.add_argument("--max-seq-length", type=int, default=be.DEFAULTS["max_seq_length"])
    p.add_argument("--max-new-tokens", type=int, default=be.DEFAULTS["max_new_tokens"])
    p.add_argument("--llm-batch-size", type=int, default=be.DEFAULTS["llm_batch_size"])
    p.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--allow-cache-config-mismatch", action="store_true", help="Explicitly allow cache model/config mismatch (not recommended)")
    return p.parse_args(argv)


def resolve_paths(args: argparse.Namespace, be) -> None:
    if args.data_dir:
        data_dir = Path(args.data_dir)
        if args.dataset and data_dir.name != args.dataset:
            raise ValueError("--dataset does not match --data-dir basename")
        dataset = args.dataset or data_dir.name
    elif args.dataset:
        dataset = args.dataset
        data_dir = Path(args.data_root) / dataset
    else:
        raise ValueError("Set --dataset CDs (recommended) or --data-dir PATH")

    args.dataset = dataset
    args.data_dir = str(data_dir)
    args.sequences = args.sequences or str(data_dir / args.sequences_file)
    args.items = args.items or str(data_dir / args.items_file)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / dataset / be.OUTPUT_SUBDIR
    args.output_dir = str(output_dir)
    args.train_output = args.train_output or str(output_dir / "train_shared.jsonl")
    args.test_output = args.test_output or str(output_dir / "test_shared.jsonl")
    args.cache_output = args.cache_output or str(output_dir / "shared_windows.cache.jsonl")

    # Prefer the frozen protocol cohort over arbitrary first-N source users.
    if not args.user_ids_file:
        protocol_users = data_dir / "protocol" / "eval_users_1000.json"
        if protocol_users.is_file():
            args.user_ids_file = str(protocol_users)
            print(f"[AUTO] Using protocol user IDs: {protocol_users}", flush=True)

    if not args.existing_train and not args.no_auto_existing_train:
        # Auto-detect only known legacy names; never assume that arbitrary JSONL is reusable.
        for candidate in (output_dir / name for name in be.LEGACY_TRAIN_NAMES):
            if candidate.is_file():
                args.existing_train = str(candidate)
                print(f"[AUTO] Found previous train cache: {candidate}", flush=True)
                break

    for field in ("sequences", "items"):
        if not Path(getattr(args, field)).is_file():
            raise FileNotFoundError(f"Missing {field} file: {getattr(args, field)}")


def verify_train_cache_metadata(path: str | None, args) -> None:
    if not path:
        return
    meta = Path(path).with_suffix(Path(path).suffix + ".meta.json")
    if not meta.is_file():
        print(f"[WARN] Missing cache metadata {meta}: model/config could not be verified.", flush=True)
        return
    obj = load_json(meta)
    mismatches = []
    checks = {"model_name": args.model_name, "max_new_tokens": args.max_new_tokens,
              "max_seq_length": args.max_seq_length, "sample": False}
    for field, expected in checks.items():
        if field in obj and obj[field] != expected:
            mismatches.append(f"{field}: previous={obj[field]!r}, requested={expected!r}")
    if mismatches:
        msg = "; ".join(mismatches)
        if not args.allow_cache_config_mismatch:
            raise ValueError("Training cache model configuration differs: " + msg +
                             ". Use --allow-cache-config-mismatch only if intentional.")
        print("[WARN] Reusing cache with mismatched model configuration: " + msg, flush=True)


def main(argv=None) -> None:
    args = parse_args(argv)
    be = BACKENDS[args.backend]
    if args.behavior_domain != "generic":
        if not hasattr(be, "set_domain"):
            raise ValueError(f"--behavior-domain is only implemented for the gemma backend")
        be.set_domain(args.behavior_domain)
        print(f"[PROMPT] behaviour extraction domain={args.behavior_domain} version={be.PROMPT_VERSION}", flush=True)
    resolve_paths(args, be)
    if args.window_size <= 0 or args.history_size < 0 or args.llm_batch_size <= 0 or args.max_attempts < 1:
        raise ValueError("window-size and batch-size must be positive, history-size non-negative")
    if args.number_of_users < 0:
        raise ValueError("number-of-users must be >= 0 (0 means all)")
    if Path(args.train_output).resolve() == Path(args.test_output).resolve():
        raise ValueError("Train and test outputs must have different paths")
    if args.existing_train and Path(args.existing_train).resolve() == Path(args.train_output).resolve():
        raise ValueError("Do not overwrite --existing-train in-place; use a different --train-output")
    if Path(args.cache_output).resolve() in (Path(args.train_output).resolve(), Path(args.test_output).resolve()):
        raise ValueError("Cache output cannot be a final train/test output")

    sequences = {str(k): v for k, v in load_json(args.sequences).items()}
    items = load_json(args.items)
    all_users = list(sequences)
    users = validate_selected_users(all_users, read_ids(args.user_ids_file, all_users), sequences)
    if args.number_of_users > 0:
        users = users[:args.number_of_users]
    if not users:
        raise ValueError("No users selected")

    verify_train_cache_metadata(args.existing_train, args)
    expected_config = be.generation_config(args)
    old_cache, read_counts = be.read_cache(args.existing_train, expected_config,
                                        args.allow_cache_config_mismatch)
    checkpoint_path = Path(args.cache_output)
    if args.existing_train and checkpoint_path.resolve() == Path(args.existing_train).resolve():
        raise ValueError("--cache-output must differ from --existing-train")
    checkpoint_cache, checkpoint_counts = be.read_cache(
        str(checkpoint_path) if checkpoint_path.is_file() else None,
        expected_config, args.allow_cache_config_mismatch)
    cache = dict(old_cache)
    cache.update(checkpoint_cache)

    # The SAME observed train windows power both output schemas.
    # NEVER read the held-out `test` items when constructing behavior.
    train_by_user: dict[str, list[Window]] = {}
    test_by_user: dict[str, list[Window]] = {}
    raw_train_items = 0
    valid_train_items = 0
    missing_train_metadata = 0
    ignored_validation_items = 0
    missing_test_field = 0
    for uid in tqdm(users, desc="Build shared train-only windows", unit="user",
                    dynamic_ncols=True, mininterval=0.5):
        user_rec = sequences[uid]
        if not isinstance(user_rec, dict):
            raise TypeError(f"User {uid!r} does not have a dict with train/test fields")
        if "train" not in user_rec:
            raise ValueError(f"User {uid!r} does not have a train history")
        train = as_ids(user_rec.get("train"))
        if not train:
            raise ValueError(f"User {uid!r} has empty train history")
        if not as_ids(user_rec.get("test")):
            missing_test_field += 1  # Not used for extraction; audit only.
        ignored_validation_items += len(as_ids(user_rec.get("val"))) + len(as_ids(user_rec.get("validation"))) + len(as_ids(user_rec.get("valid")))
        raw_train_items += len(train)
        capped_train = train[-args.history_size:] if args.history_size > 0 else train
        valid_train_items += sum(get_info(items, item) is not None for item in capped_train)
        missing_train_metadata += sum(get_info(items, item) is None for item in capped_train)
        train_windows = make_train_windows(uid, train, items, args.history_size, args.window_size)
        if not train_windows:
            raise ValueError(f"User {uid!r} has no titled observed train items")
        train_by_user[uid] = train_windows
        test_by_user[uid] = train_windows  # identical history/windows, no validation

    if ignored_validation_items:
        print(f"[WARN] Ignoring {ignored_validation_items} validation interactions present in input; this protocol uses train only.", flush=True)
    if missing_test_field:
        print(f"[WARN] {missing_test_field} users have no test item; target labels were never used in precompute.", flush=True)

    train_all = [w for uid in users for w in train_by_user[uid]]
    test_all = [w for uid in users for w in test_by_user[uid]]
    needed_windows: dict[str, Window] = {}
    for w in train_all + test_all:
        needed_windows.setdefault(window_key(w), w)
    in_train = {window_key(w) for w in train_all}
    in_test = {window_key(w) for w in test_all}
    if in_train != in_test or len(needed_windows) != len(in_train):
        raise AssertionError("Train/test behavior windows differ in no-validation mode")
    missing = {key: win for key, win in needed_windows.items() if key not in cache}

    info = {
        "dataset": args.dataset, "sequences": args.sequences, "items": args.items,
        "train_output": args.train_output, "test_output": args.test_output,
        "users": len(users), "same_users_train_test": True,
        "same_observed_train_history": True, "validation_used": False,
        "test_ground_truth_used_in_extraction": False,
        "ignored_validation_items_present_in_source": ignored_validation_items,
        "users_without_test_target": missing_test_field,
        "user_ids_file": args.user_ids_file,
        "raw_train_items": raw_train_items,
        "valid_train_items_after_history_cap": valid_train_items,
        "skipped_train_items_without_title_after_cap": missing_train_metadata,
        "history_size": args.history_size, "window_size": args.window_size,
        "test_window_policy": "same_as_train_no_validation",
        "train_windows": len(train_all), "test_windows": len(test_all),
        "shared_train_test_window_keys": len(in_train & in_test),
        "test_only_window_keys": len(in_test - in_train),
        "train_only_window_keys": len(in_train - in_test),
        "total_unique_windows": len(needed_windows),
        "valid_existing_train_cache_rows": read_counts.get("valid_records", 0),
        "valid_checkpoint_cache_rows": checkpoint_counts.get("valid_records", 0),
        "checkpoint_config_mismatch_skipped": checkpoint_counts.get("generation_config_mismatch", 0),
        "windows_reused_from_cache": len(needed_windows) - len(missing),
        "llm_windows_pending": len(missing),
        "llm_generate_calls_pending_at_batch_size": (len(missing) + args.llm_batch_size - 1) // args.llm_batch_size,
        "cache_checkpoint_path": str(checkpoint_path),
        "max_new_tokens": args.max_new_tokens,
        "backend": args.backend, "prompt_version": be.PROMPT_VERSION,
    }
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    for output in [args.train_output, args.test_output]:
        if Path(output).exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite or another filename")

    stats = be.infer_missing(missing, cache, args, checkpoint_path)
    records: list[dict] = []
    for user_order, uid in enumerate(
            tqdm(users, desc="Pack train records", unit="user", dynamic_ncols=True,
                 mininterval=0.5)):
        for i, w in enumerate(train_by_user[uid]):
            records.append(to_train_record(w, cache[window_key(w)], user_order, i, len(records), be.PROMPT_VERSION))
    write_jsonl_atomic(args.train_output, records, args.overwrite,
                       progress_label="Write train JSONL")
    print(f"[WRITE] train records={len(records)} users={len(users)} path={args.train_output}", flush=True)

    test_results = []
    for uid in tqdm(users, desc="Pack test records", unit="user",
                    dynamic_ncols=True, mininterval=0.5):
        observed = as_ids(sequences[uid].get("train"))  # train ONLY; never val/test
        if args.history_size > 0:
            observed = observed[-args.history_size:]
        test_results.append(to_test_row(uid, observed, test_by_user[uid], cache,
                                        args.recent_behaviors, args.window_size,
                                        args.history_size, "same_as_train_no_validation",
                                        be.PROMPT_VERSION))
    write_jsonl_atomic(args.test_output, test_results, args.overwrite,
                       progress_label="Write test JSONL")
    print(f"[WRITE] test users={len(test_results)} path={args.test_output}", flush=True)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
