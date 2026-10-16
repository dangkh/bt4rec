#!/usr/bin/env python3
"""End-to-end equivalence check: legacy scripts vs the refactored bt4rec package.

Runs the whole pipeline twice on a synthetic dataset (no GPU / LLM / network):
  protocol -> precompute (pre-filled LLM cache) -> build tree -> inference --prompts-only
once with the legacy scripts and once with ``python -m bt4rec.cli.*`` in
legacy-compatible settings (--label-assignment kmeans --state-signature mode),
then asserts that every artefact is identical. Finally runs the refactored
pipeline with the new defaults to make sure it works end to end.

Usage:
    python scripts/dev/verify_against_legacy.py --legacy-dir legacy --workdir /tmp/bt4rec_verify
Needs numpy, scikit-learn and tqdm (torch / unsloth are stubbed if missing).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fixtures.synthetic import fake_behavior, write_dataset  # noqa: E402

from bt4rec.behavior import gemma  # noqa: E402
from bt4rec.behavior.windows import cache_key, make_train_windows  # noqa: E402

MODES = ("native,latest_only,latest_how,random_how,global_continuation,tree,tree_original,"
         "recent3,recent3_tree,stay_only,tree_stay,recent3_tree_stay")

PATCH_RUNNER = textwrap.dedent('''
    import importlib.util, runpy, sys
    sys.path[:0] = {paths!r}
    from fixtures.synthetic import FakeEncoder, FakeRanker, hash_embed
    target, kind = sys.argv[1], sys.argv[2]
    sys.argv = [target] + sys.argv[3:]
    if target.endswith('.py'):
        spec = importlib.util.spec_from_file_location('legacy_mod', target)
        mod = importlib.util.module_from_spec(spec); sys.modules['legacy_mod'] = mod; spec.loader.exec_module(mod)
        if kind == 'tree':
            mod.encode_qwen = lambda texts, *a, **k: hash_embed(texts)
        if kind == 'infer':
            mod.QwenEncoder = FakeEncoder
            parse = mod.LocalQwenRanker._parse_rank
            mod.LocalQwenRanker = lambda *a, **k: FakeRanker(parse_fn=parse)
        mod.main()
    else:
        import bt4rec.tree.embeddings as emb, bt4rec.tree.vocab as vocab, bt4rec.cli.run as run
        emb.encode_qwen = lambda texts, *a, **k: hash_embed(texts)
        vocab.QwenEncoder = FakeEncoder
        run.QwenEncoder = FakeEncoder
        import bt4rec.llm.ranker as rk
        run.LocalQwenRanker = lambda *a, **k: FakeRanker(parse_fn=rk.LocalQwenRanker._parse_rank)
        mod = importlib.import_module(target)
        mod.main()
''')


def sh(cmd, cwd, env):
    r = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:], sep="\n")
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd))}")
    return r.stdout


def prefill_cache(work: Path, users, history=20, ws=3):
    ds = work / "data" / "CDs"
    seqs = json.loads((ds / "user_sequences.json").read_text())
    items = json.loads((ds / "items.json").read_text())
    args = argparse.Namespace(**{k: v for k, v in gemma.DEFAULTS.items()}, load_in_4bit=True, dtype="auto")
    cfg = gemma.generation_config(args)
    out = work / "precomputed" / "CDs"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "shared_windows.cache.jsonl").open("w") as f:
        for uid in users:
            for w in make_train_windows(uid, seqs[uid]["train"], items, history, ws):
                b = fake_behavior(uid, w.items)
                rec = {"schema_version": "amem_local_memory_dual_behavior_v4",
                       "behavior_prompt_version": gemma.PROMPT_VERSION, "user_id": uid,
                       "interaction_sequence": w.items, "generation_config": cfg, **b,
                       "behavior_signature": b["trajectory_signature"], "consistency_adjustments": [],
                       "structural_evidence": {}, "pattern_description": b["recommendation_behavior"],
                       "keywords": b["semantic_focus"], "parse_ok": True, "parse_error": None,
                       "raw_response": "{}"}
                assert cache_key(uid, w.items)
                f.write(json.dumps(rec) + "\n")


def same_json(a: Path, b: Path, drop=("metadata",)):
    def load(p):
        if p.suffix == ".jsonl":
            return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        obj = json.loads(p.read_text())
        if isinstance(obj, dict):
            for k in drop:
                obj.pop(k, None)
        return obj
    def norm(x):  # tolerate last-bit float noise from set-ordered summation
        if isinstance(x, float):
            return round(x, 9)
        if isinstance(x, dict):
            return {k: norm(v) for k, v in x.items()}
        if isinstance(x, list):
            return [norm(v) for v in x]
        return x
    return norm(load(a)) == norm(load(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-dir", required=True)
    ap.add_argument("--workdir", default="/tmp/bt4rec_verify")
    a = ap.parse_args()
    legacy = Path(a.legacy_dir).resolve()
    work = Path(a.workdir)
    shutil.rmtree(work, ignore_errors=True)
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    stubs = os.environ.get("BT4REC_STUBS", "")
    paths = [str(ROOT / "src"), str(ROOT / "tests")] + ([stubs] if stubs else [])
    env["PYTHONPATH"] = os.pathsep.join(paths + [env.get("PYTHONPATH", "")])
    runner = work / "runner.py"
    work.mkdir(parents=True)
    runner.write_text(PATCH_RUNNER.format(paths=paths))
    py = sys.executable
    results = {}

    for side in ("legacy", "new"):
        w = work / side
        write_dataset(w)
        if side == "legacy":
            sh([py, legacy / "prepare_data.py", "--dataset", "CDs", "--sizes", "20,60", "--seed", "42"], w, env)
        else:
            sh([py, "-m", "bt4rec.cli.prepare_protocol", "--dataset", "CDs", "--sizes", "20,60", "--seed", "42"], w, env)
        users = json.loads((w / "data/CDs/protocol/eval_users_60.json").read_text())
        prefill_cache(w, users)
        pre = ["--dataset", "CDs", "--user-ids-file", "data/CDs/protocol/eval_users_60.json",
               "--history-size", "20", "--window-size", "3"]
        if side == "legacy":
            sh([py, legacy / "precompute_behavior.py", *pre], w, env)
        else:
            sh([py, "-m", "bt4rec.cli.precompute", "--backend", "gemma", *pre], w, env)
        tree = ["--input", "precomputed/CDs/train_shared.jsonl", "--output-dir", "trees/CDs_60_20",
                "--num-clusters", "15", "--min-support-users", "2", "--seed", "42", "--device", "cpu"]
        if side == "legacy":
            sh([py, runner, str(legacy / "build_reverse_behavior_tree.py"), "tree", *tree], w, env)
        else:
            sh([py, runner, "bt4rec.cli.build_tree", "tree", *tree,
                "--label-assignment", "kmeans", "--state-signature", "mode"], w, env)
        infer = ["--dataset", "CDs", "--tree", "trees/CDs_60_20", "--data-root", "data",
                 "--precomputed-behaviors", "precomputed/CDs/test_shared.jsonl",
                 "--train-behaviors", "precomputed/CDs/train_shared.jsonl",
                 "--user-ids-file", "data/CDs/protocol/eval_users_20.json",
                 "--candidate-file", "data/CDs/protocol/user_candidates_testpool1000_seed42.json",
                 "--source-user-ids-file", "data/CDs/protocol/train_users_60.json",
                 "--history-size", "20", "--max-users", "20", "--modes", MODES,
                 "--save-all-prompts", "--run-dir", "results/run",
                 "--embedding-device", "cpu", "--allow-tree-source-mismatch"]
        infer_new = infer + ["--stay-prompt", "v1"]
        if side == "legacy":
            sh([py, runner, str(legacy / "inference_bt4rec_ablation.py"), "infer", *infer], w, env)
        else:
            sh([py, runner, "bt4rec.cli.run", "infer", *infer_new], w, env)
        results[side] = w

    L, N = results["legacy"], results["new"]
    checks = {
        "protocol/eval_users_60": "data/CDs/protocol/eval_users_60.json",
        "protocol/candidates": "data/CDs/protocol/user_candidates_testpool1000_seed42.json",
        "precompute/train_shared": "precomputed/CDs/train_shared.jsonl",
        "precompute/test_shared": "precomputed/CDs/test_shared.jsonl",
        "tree/behavior_states(states)": "trees/CDs_60_20/behavior_states.json",
        "tree/behavior_tree(nodes)": "trees/CDs_60_20/behavior_tree.json",
        "inference/prompts_all": "results/run/prompts_all.jsonl",
        "inference/rankings": "results/run/rankings.jsonl",
    }
    ok = True
    for name, rel in checks.items():
        same = same_json(L / rel, N / rel)
        if name.startswith("tree/behavior_states"):
            sl = json.loads((L / rel).read_text())["states"]
            sn = json.loads((N / rel).read_text())["states"]
            extra = {"signature_mode", "mode_signature", "mode_signature_count", "medoid_signature"}
            same = [{k: v for k, v in s.items() if k not in extra} for s in sn] == sl
        print(f"{'OK  ' if same else 'DIFF'} {name}")
        ok &= same
    sl = json.loads((L / "results/run/summary.json").read_text())
    sn = json.loads((N / "results/run/summary.json").read_text())
    same = sl["metrics"] == sn["metrics"] and sl["metrics_complete_only"] == sn["metrics_complete_only"]
    print(f"{'OK  ' if same else 'DIFF'} inference/summary metrics")
    ok &= same
    import numpy as np
    e = np.allclose(np.load(L / "trees/CDs_60_20/behavior_state_embeddings.npy"),
                    np.load(N / "trees/CDs_60_20/behavior_state_embeddings.npy"))
    print(f"{'OK  ' if e else 'DIFF'} tree/state_embeddings")
    ok &= e

    # New defaults (cosine relabel + medoid signature) must also run end to end.
    w = work / "new_defaults"
    shutil.copytree(N, w)
    shutil.rmtree(w / "trees")
    shutil.rmtree(w / "results")
    sh([py, runner, "bt4rec.cli.build_tree", "tree", "--input", "precomputed/CDs/train_shared.jsonl",
        "--output-dir", "trees/CDs_60_20", "--num-clusters", "15", "--min-support-users", "2",
        "--seed", "42", "--device", "cpu"], w, env)
    sh([py, runner, "bt4rec.cli.run", "infer", *infer], w, env)
    meta = json.loads((w / "trees/CDs_60_20/behavior_states.json").read_text())["metadata"]
    print("new defaults:", meta["cluster_info"].get("label_assignment"),
          "labels_changed_from_kmeans =", meta["cluster_info"].get("labels_changed_from_kmeans"),
          "| prompts:", sum(1 for _ in open(w / "results/run/prompts_all.jsonl")), "users")
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
