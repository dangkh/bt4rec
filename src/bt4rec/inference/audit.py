from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from bt4rec.common.io import dump_json, file_sha256, load_json, load_user_ids_file
from bt4rec.inference.constants import (
    RANK_PROMPT_VERSION,
    RECENT3_PROMPT_VERSION,
    SCHEMA_VERSION,
    STAY_MODES,
    STAY_PROMPT_VERSIONS,
)
from bt4rec.ranking.prompts import build_rank_prompt, build_recent3_rank_prompt


def validate_tree_provenance(args: argparse.Namespace, cohort: Sequence[str]) -> Dict[str, Any]:
    """Audit Tree origin and user coverage without treating the Tree as a user lookup.

    The collaborative Tree is a learned behavior-state transition model, not an
    index keyed by evaluation user IDs.  A target user with a valid behavior
    cache may map to the existing states even if absent from the training-user
    selection recorded in ``selected_user_order``.  This is different from
    missing frozen candidates/targets/behavior caches, which remain fatal.

    Always report train/eval cohort overlap: high overlap is *not* compatible
    with a user-disjoint evaluation claim, even though train-only interactions
    need not imply held-out test-label leakage.
    """
    metadata = load_json(args.states_json).get("metadata", {})
    input_path = metadata.get("input")
    source_matches = bool(input_path and Path(input_path).resolve() == Path(args.expected_tree_input).resolve())
    warnings: List[str] = []
    if input_path and not source_matches:
        msg = (f"Tree was built from {input_path!r}, but expected {args.expected_tree_input!r}. "
               "Verify that Tree and test cache use compatible behavior schema, model, and split.")
        if not args.allow_tree_source_mismatch:
            raise ValueError(msg + " Use --allow-tree-source-mismatch only after verifying provenance.")
        warnings.append(msg)
    if not input_path:
        warnings.append("Tree state metadata lacks original input path; provenance cannot be verified.")

    tree_users = metadata.get("selected_user_order")
    eval_ids = set(map(str, cohort))
    num_tree_users: Optional[int] = None
    overlap_count: Optional[int] = None
    missing_from_tree: List[str] = []
    if isinstance(tree_users, list):
        training_ids = set(map(str, tree_users))
        num_tree_users = len(training_ids)
        overlap_count = len(eval_ids & training_ids)
        missing_from_tree = sorted(eval_ids - training_ids)
        print(
            f"[TREE USER AUDIT] eval={len(eval_ids)} "
            f"present_in_tree_training={overlap_count} "
            f"absent_from_tree_training={len(missing_from_tree)}; "
            f"examples={missing_from_tree[:5]}", flush=True,
        )
        if missing_from_tree:
            msg = (
                f"Tree training metadata omits {len(missing_from_tree)} of "
                f"{len(eval_ids)} eval users; examples={missing_from_tree[:5]}. "
                "This is allowed: Tree predictions use mapped behavior states "
                "rather than a lookup by training user ID."
            )
            if getattr(args, "require_eval_users_in_tree", False):
                raise ValueError(msg + " Disable --require-eval-users-in-tree to evaluate unseen users.")
            warnings.append(msg)
        if overlap_count:
            warnings.append(
                f"Train-user/eval-user overlap: {overlap_count}/{len(eval_ids)} users. "
                "Do not describe this run as user-disjoint unless you use a disjoint Tree."
            )
    else:
        warnings.append("Tree lacks selected_user_order metadata; train/eval user overlap cannot be audited.")

    for warning in warnings:
        print(f"[TREE PROVENANCE WARN] {warning}", flush=True)
    return {
        "tree_training_source": input_path,
        "tree_matches_expected_qwen_cache": source_matches,
        "tree_users": num_tree_users,
        "eval_users": len(eval_ids),
        "eval_users_present_in_tree_training": overlap_count,
        "eval_users_absent_from_tree_training": len(missing_from_tree) if isinstance(tree_users, list) else None,
        "eval_users_absent_examples": missing_from_tree[:10],
        "user_disjoint_tree_cohort": (overlap_count == 0) if overlap_count is not None else None,
        "warnings": warnings,
    }


def audit_profile_source_restrictions(args, users, sequences, modes):
    allowed = set(load_user_ids_file(args.source_user_ids_file)) if args.source_user_ids_file else None
    if allowed is not None and not allowed:
        raise ValueError('Source cohort file empty')
    full_eval = set(load_user_ids_file(args.user_ids_file))
    forbidden = full_eval if args.exclude_all_eval_sources else set()
    if {'random_how','random_only'} & set(modes) or 'pmn' in modes:
        print('[SOURCE] train_cohort=', len(allowed) if allowed is not None else 'unrestricted',
              ' eval_excluded=',len(forbidden), ' source/eval overlap prohibited self-match always',flush=True)
    return allowed, forbidden


def manifest_checkpoint(args, modes, users):
    files = {
        'candidate_file':args.candidate_file, 'behavior_cache':args.precomputed_behaviors,
        'tree_json':args.tree_json, 'states_json':args.states_json,
        'sequences':args.sequences, 'user_ids_file':args.user_ids_file,
        'train_behaviors':args.train_behaviors if {'random_how','random_only'} & set(modes) else None,
        'pmn_summaries':args.pmn_summaries if 'pmn' in modes else None,
        'oracle_behaviors':args.oracle_behaviors if 'oracle' in modes else None,
        'source_user_ids_file':args.source_user_ids_file,
    }
    return {'schema':SCHEMA_VERSION,'prompt_version':RANK_PROMPT_VERSION,'dataset':args.dataset,
            'users':users,'modes':modes, 'history_size':args.history_size,
            'tree_max_query_order':args.tree_max_query_order,'top_next':args.top_next,
            'model':args.model,'pmn_embedding_model':args.pmn_embedding_model,
            'llm_batch_size':args.llm_batch_size,
            'rank_max_new_tokens':args.rank_max_new_tokens,'temperature':args.temperature,
            'candidate_size':args.candidate_size,'seed':args.seed,
            'max_behaviors':args.max_behaviors,'respect_mapping_filter':args.respect_mapping_filter,
            'mapping_min_similarity':args.mapping_min_similarity,'mapping_min_margin':args.mapping_min_margin,
            'cluster_text_field':args.cluster_text_field,'embedding_model':args.embedding_model,
            'embedding_batch_size':args.embedding_batch_size,'max_seq_length':args.max_seq_length,
            'load_in_4bit':args.load_in_4bit,'strict_cache_history':args.strict_cache_history,
            'exclude_all_eval_sources':args.exclude_all_eval_sources,
            'require_user_disjoint_tree':args.require_user_disjoint_tree,
            'stay_prompt_version':(STAY_PROMPT_VERSIONS[getattr(args, 'stay_prompt', 'v2')]
                                   if set(modes) & STAY_MODES else None),
            'files':{k:str(Path(v).resolve()) if v else None for k,v in files.items()},
            'external_file_hashes':{
                k:file_sha256(files[k]) for k in ('train_behaviors','pmn_summaries',
                    'oracle_behaviors','source_user_ids_file') if files[k] is not None},
            }


def filled_prompt_sections(record: Dict[str, Any], run_native: bool, run_latest: bool,
                            run_recent3: bool = False) -> Dict[str, str]:
    h, c, latest, trans = (record["history_items"], record["candidate_items"],
                            record["latest_behavior"], record["predicted_transition"])
    sections = {"tree": build_rank_prompt(history_items=h, candidate_items=c,
                                           latest_behavior=latest, predicted_transition=trans)}
    if run_latest:
        sections["latest_only"] = build_rank_prompt(history_items=h, candidate_items=c,
                                                     latest_behavior=latest, predicted_transition=None)
    if run_native:
        sections["native"] = build_rank_prompt(history_items=h, candidate_items=c,
                                               latest_behavior=None, predicted_transition=None)
    if run_recent3:
        sections["recent3"] = build_recent3_rank_prompt(
            history_items=h, candidate_items=c,
            recent3_behaviors=record["recent3_behaviors"],
        )
    return sections


def save_prompt_examples(args: argparse.Namespace, prepared: List[Dict[str, Any]], run_dir: Path) -> Dict[str, Any]:
    if not prepared:
        raise ValueError("Nothing prepared to save prompts")
    chosen = prepared[0]
    if args.prompt_example_user:
        hits = [r for r in prepared if r["user_id"] == args.prompt_example_user]
        if not hits:
            raise ValueError(f"--prompt-example-user={args.prompt_example_user!r} not selected for inference")
        chosen = hits[0]
    sections = filled_prompt_sections(chosen, args.run_native, args.run_latest_only,
                                        args.run_recent3)
    system_msg = "You are a recommendation ranking system. Return only the requested candidate-label ranking."
    text_parts = [f"Dataset: {args.dataset}", f"Example user: {chosen['user_id']}",
                  "System message (common to every condition):", system_msg]
    for mode, prompt in sections.items():
        text_parts += ["", "="*32 + f" {mode.upper()} " + "="*32, prompt]
    (run_dir / "prompt_example.txt").write_text("\n".join(text_parts) + "\n", encoding="utf-8")
    details = {
        "dataset": args.dataset, "user_id": chosen["user_id"],
        "system_message": system_msg, "rank_prompt_version": RANK_PROMPT_VERSION,
        "history_items": chosen["history_items"],
        "latest_observed_behavior": chosen["latest_behavior"],
        "recent3_observed_behaviors": chosen["recent3_behaviors"] if args.run_recent3 else None,
        "recent3_prompt_version": RECENT3_PROMPT_VERSION if args.run_recent3 else None,
        "tree_query": chosen["tree_result"],
        "predicted_next_states": chosen["predicted_transition"],
        "candidate_label_map": {f"C{i:02d}": x for i,x in enumerate(chosen["candidate_items"],1)},
        "prompts": sections,
        "notice": "Held-out test item is NOT identified in the model prompt. Targets are used only for metrics.",
    }
    dump_json(run_dir / "prompt_example.json", details)
    if args.save_all_prompts:
        with (run_dir / "prompts_all.jsonl").open("w", encoding="utf-8") as out:
            for r in prepared:
                out.write(json.dumps({"user_id": r["user_id"],
                                      "prompts": filled_prompt_sections(r, args.run_native, args.run_latest_only,
                                                                         args.run_recent3)},
                                     ensure_ascii=False) + "\n")
    return {"sample_user_id": chosen["user_id"], "prompt_example_txt": str(run_dir / "prompt_example.txt"),
            "prompt_example_json": str(run_dir / "prompt_example.json"),
            "all_prompts_jsonl": str(run_dir / "prompts_all.jsonl") if args.save_all_prompts else None}
