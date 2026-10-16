"""Packing of train/test behavior records. Backend-independent: the prompt
version is passed in by the caller (behavior/gemma.py or behavior/qwen.py)."""
import argparse
import hashlib
import json
from typing import Any

from bt4rec.behavior.windows import Window, compact_test_window, window_key

SCHEMA_TRAIN = "amem_local_memory_dual_behavior_v4"


SCHEMA_TEST = "amem_test_behavior_precompute_dual_v3"


def generation_config(args: argparse.Namespace, prompt_version: str) -> dict:
    """Fields that must match before new checkpoint behavior reuse."""
    return {"model_name": args.model_name, "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens, "load_in_4bit": args.load_in_4bit,
            "dtype": args.dtype, "sample": False, "prompt_version": prompt_version}


def to_train_record(w: Window, rec: dict, user_order: int, index: int, pcid: int,
                    prompt_version: str) -> dict:
    output = dict(rec)
    output.update({
        "schema_version": SCHEMA_TRAIN, "behavior_prompt_version": prompt_version,
        "precompute_id": pcid, "user_id": w.user_id,
        "user_order": user_order, "window_index": index,
        "interaction_sequence": w.items,
    })
    return output


def to_test_behavior(rec: dict, uid: str, i: int, prompt_version: str) -> dict:
    keys = ["behavior_explanation", "recommendation_behavior", "semantic_focus",
            "trajectory_signature", "behavior_signature", "mechanism", "scope",
            "direction", "confidence", "consistency_adjustments", "structural_evidence",
            "pattern_description", "keywords"]
    out = {k: rec.get(k) for k in keys}
    out.update(task_id=f"{uid}_W{i}", window_index=i,
               behavior_prompt_version=prompt_version)
    return out


def to_test_row(uid: str, obs: list[Any], windows: list[Window], cache: dict,
                recent_count: int, size: int, cap: int, mode: str,
                prompt_version: str) -> dict:
    b = [to_test_behavior(cache[window_key(w)], uid, i, prompt_version) for i, w in enumerate(windows)]
    evidence_fields = ["window_index", "recommendation_behavior", "semantic_focus",
                       "trajectory_signature", "behavior_signature", "mechanism", "scope",
                       "direction", "confidence"]
    evidence = [{k: bh.get(k) for k in evidence_fields} for bh in (b[-recent_count:] if recent_count > 0 else [])]
    return {
        "schema_version": SCHEMA_TEST,
        "behavior_prompt_version": prompt_version,
        "behavior_schema": "dual_view_semantic_plus_trajectory",
        "window_policy": mode,
        "precompute_ok": True,
        "user_id": uid,
        "train_history_item_ids": [str(x) for x in obs],
        "train_history_hash": hashlib.sha256(json.dumps([str(x) for x in obs], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "window_size": size,
        "max_train_interactions": cap,
        "behavior_windows": [compact_test_window(w) for w in windows],
        "generated_behaviors": b,
        "recent_behavior_evidence": evidence,
    }
