from __future__ import annotations

import argparse
from pathlib import Path

from bt4rec.inference.constants import RANDOM_SOURCE_MODES


def canonical_dataset(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"cd":"CDs","cds":"CDs","video":"Video_Games","vg":"Video_Games",
               "videogames":"Video_Games","video_games":"Video_Games",
               "ml":"ML","ml1m":"ML","movielens":"ML","movielens_1m":"ML",
               "books":"Books","book":"Books","software":"Software","soft":"Software"}
    out = aliases.get(key, value.strip())
    if not out or out in (".", "..") or "/" in out or "\\" in out or ".." in out:
        raise ValueError(f"Unsafe dataset name {value!r}")
    return out


def resolve_tree_directory(value: str) -> Path:
    """Explicit override, including a bare folder name under ./trees/."""
    supplied = Path(value).expanduser()
    candidates = [supplied]
    if len(supplied.parts) == 1:
        candidates += [Path("trees") / supplied, Path("behavior_trees") / supplied]
    existing = [x for x in candidates if x.is_dir()]
    if not existing:
        raise FileNotFoundError("Tree directory not found. Tried: " + ", ".join(map(str, candidates)))
    if len(existing) > 1 and len({x.resolve() for x in existing}) > 1:
        raise ValueError("Ambiguous Tree name, use its explicit path: " + ", ".join(map(str, existing)))
    return existing[0]


def auto_tree_dir(dataset: str, tree_root: str) -> Path:
    """Known dataset Tree folders; require an override rather than guessing ambiguously."""
    tree_names = {
        "ML": "ML_1000_20",
        "CDs": "CDs_1000_20",
        "Video_Games": "VGs_1000_20",
        "Books": "Books_1000_20",
        "Software": "Software_1000_20",
    }
    root = Path(tree_root).expanduser()
    preferred = root / tree_names.get(dataset, f"{dataset}_1000_20")
    if preferred.is_dir():
        return preferred

    # If preferred folder is absent, search only unambiguous trees for the same
    # dataset. Never silently select a tree trained for another dataset.
    stems = {
        "ML": ["ML_", "ML1M_", "tree_ML_"],
        "CDs": ["CDs_", "tree_CDs_"],
        "Video_Games": ["VGs_", "Video_Games_", "tree_VGs_", "tree_Video_Games_"],
        "Books": ["Books_", "tree_Books_"],
        "Software": ["Software_", "tree_Software_"],
    }.get(dataset, [dataset + "_", "tree_" + dataset + "_"])
    matches = []
    if root.is_dir():
        for folder in sorted(root.iterdir()):
            if folder.is_dir() and any(folder.name.startswith(prefix) for prefix in stems):
                if all((folder / name).is_file() for name in (
                    "behavior_states.json", "behavior_state_embeddings.npy", "behavior_tree.json"
                )):
                    matches.append(folder)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Found multiple Trees for {dataset}: {', '.join(map(str, matches))}. "
            "Choose explicitly using --tree."
        )
    raise FileNotFoundError(
        f"No Tree found for dataset {dataset} under {root}. "
        f"Expected {preferred}. Pass --tree /path/to/tree to override."
    )


def resolve_inputs(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve standardized dataset input paths without changing the ranking method."""
    args.dataset = canonical_dataset(args.dataset)
    args.tree_dir = str(
        resolve_tree_directory(args.tree_dir)
        if args.tree_dir
        else auto_tree_dir(args.dataset, args.tree_root)
    )
    if args.history_size is None:
        args.history_size = 30 if args.dataset == "ML" else 20
    if args.max_train_interactions is None:
        args.max_train_interactions = args.history_size
    for name in ("history_size", "llm_batch_size", "candidate_size", "top_next"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    for name in ("max_users", "max_behaviors", "start_user", "tree_max_query_order"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be >=0")

    dataset_dir = Path(args.data_root) / args.dataset
    protocol_dir = dataset_dir / "protocol"
    behavior_root = Path(args.precompute_root) / args.dataset
    default_cache = behavior_root / "test_shared.jsonl"
    legacy_cache = behavior_root / args.behavior_model / "test_shared.jsonl"
    # Canonical data has no model-specific subfolder. Legacy fallback is used
    # only if the canonical file does not exist; explicit overrides take priority.
    if not default_cache.is_file() and legacy_cache.is_file():
        default_cache = legacy_cache

    default_inputs = {
        "items": dataset_dir / "items.json",
        "sequences": dataset_dir / "user_sequences.json",
        "negatives": dataset_dir / "user_negatives.json",
        "candidate_file": protocol_dir / "user_candidates_testpool1000_seed42.json",
        "user_ids_file": protocol_dir / "eval_users_1000.json",
        "precomputed_behaviors": default_cache,
    }
    for name, path in default_inputs.items():
        if getattr(args, name) is None:
            setattr(args, name, str(path))

    # Use the train cache in the *same actual folder* as the test cache to
    # prevent false provenance mismatches from the obsolete /qwen/ assumption.
    args.expected_tree_input = str(Path(args.precomputed_behaviors).parent / "train_shared.jsonl")

    for name in default_inputs:
        path = Path(getattr(args, name))
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {name}: {path}. Override with --{name.replace('_','-')} "
                "or check the standardized dataset directory."
            )
    states_json = args.states_json or str(Path(args.tree_dir) / "behavior_states.json")
    states_embeddings = args.state_embeddings or str(Path(args.tree_dir) / "behavior_state_embeddings.npy")
    tree_json = args.tree_json or str(Path(args.tree_dir) / "behavior_tree.json")
    for name, path in (("states", states_json), ("embeddings", states_embeddings), ("Tree", tree_json)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing {name} artifact: {path}")
    args.states_json = states_json
    args.state_embeddings = states_embeddings
    args.tree_json = tree_json
    return args


def resolve_source_files(args, modes):
    need_random = bool(RANDOM_SOURCE_MODES & set(modes))
    if args.train_behaviors is None and need_random:
        args.train_behaviors = str(Path(args.precomputed_behaviors).parent/'train_shared.jsonl')
    if args.pmn_summaries is None and 'pmn' in modes:
        args.pmn_summaries = str(Path(args.precomputed_behaviors).parent/'pmn_preference_summaries_gemma_v4.jsonl')
    if args.oracle_behaviors is None and 'oracle' in modes:
        args.oracle_behaviors = str(Path(args.precomputed_behaviors).parent/'target_user_oracle_protocol_seed42.jsonl')
    if args.source_user_ids_file is None:
        for guess in (Path(args.data_root)/args.dataset/'protocol'/'train_users_1000.json',
                      Path(args.data_root)/args.dataset/'train_users_1000.json'):
            if guess.is_file():
                args.source_user_ids_file = str(guess)
                break
    used = [('train-behaviors',args.train_behaviors,need_random,'random_how_only'),
            ('pmn-summaries',args.pmn_summaries,'pmn' in modes,'pmn'),
            ('oracle-behaviors',args.oracle_behaviors,'oracle' in modes,'oracle')]
    for name,path,needed,mode in used:
        if needed and (not path or not Path(path).is_file()):
            raise FileNotFoundError(f'--{name} required for {mode}: {path}')
    if need_random and Path(args.train_behaviors).resolve()==Path(args.precomputed_behaviors).resolve():
        raise ValueError('Random HOW training stream MUST differ from target test behavior cache')
