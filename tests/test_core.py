"""Fast unit tests (CPU only, no LLM). Run: python -m pytest tests  (or python -m unittest)."""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

import numpy as np

from bt4rec import taxonomy as tx
from bt4rec.behavior.windows import chunk_windows
from bt4rec.eval.metrics import metrics_from_ranks, paired_bootstrap
from bt4rec.ranking.parse import sanitize_ranking
from bt4rec.tree.assign import CompatibilityIndex, nearest_compatible_state, relabel_by_cosine
from bt4rec.tree.builder import (
    accumulate_contexts,
    build_context_records,
    build_reverse_tree,
    build_user_sequences,
    validate_tree_structure,
)
from bt4rec.tree.query import latest_accepted_state_suffix


class TaxonomyTest(unittest.TestCase):
    def test_punctuation_insensitive(self):
        self.assertEqual(tx.normalize_scope("same collection/series"), "same series")
        self.assertEqual(tx.normalize_scope("Same-Collection Series"), "same series")
        self.assertEqual(tx.normalize_mechanism("cross-category exploration"), "cross category exploration")
        self.assertEqual(tx.normalize_mechanism("Persistence."), "persistence")
        self.assertEqual(tx.normalize_direction("Deepening"), "deepen")

    def test_cluster_text_same_for_builder_and_inference(self):
        # Before the refactor the builder kept '/' -> different embedded text at test time.
        raw = {"mechanism": "Collection Expansion", "scope": "same collection/series",
               "direction": "deepen", "trajectory_signature": "  moves  deeper "}
        alias = dict(raw, scope="same collection series", trajectory_signature=None,
                     behavior_signature="moves deeper")
        text = tx.choose_cluster_text(raw, "structured_combined")
        self.assertEqual(text, "mechanism=collection expansion; scope=same series; "
                               "direction=deepen; trajectory=moves deeper")
        self.assertEqual(text, tx.choose_cluster_text(alias, "structured_combined"))

    def test_normalize_choice(self):
        self.assertEqual(tx.normalize_choice("cross category exploration", tx.MECHANISMS, "mechanism"),
                         "cross-category exploration")
        with self.assertRaises(ValueError):
            tx.normalize_choice("teleportation", tx.MECHANISMS, "mechanism")


class AssignTest(unittest.TestCase):
    def test_acceptance_rules(self):
        scores = np.array([0.9, 0.89, 0.2, 0.95])
        a = nearest_compatible_state(scores, [0, 1, 2], 0.55, 0.02)
        self.assertEqual((a.best_index, a.second_index), (0, 1))
        self.assertFalse(a.accepted)
        self.assertEqual(a.reasons, ["low_margin"])
        b = nearest_compatible_state(scores, [2], 0.55, 0.02)  # single-state bucket: no margin
        self.assertEqual(b.reasons, ["low_similarity"])
        c = nearest_compatible_state(scores, [0, 3], 0.55, 0.02)
        self.assertTrue(c.accepted)
        self.assertEqual(c.best_index, 3)

    def test_compatibility_index(self):
        rows = [{"behavior_bucket_key": ["persistence"]}, {"behavior_bucket_key": ["Shifting"]},
                {"behavior_bucket_key": ["persistence"]}]
        idx = CompatibilityIndex(rows, "constrained", "mechanism")
        self.assertEqual(idx.candidates({"mechanism": "Persistence"}), ([0, 2], True))
        self.assertEqual(idx.candidates({"mechanism": "unknown"}), ([0, 1, 2], False))
        g = CompatibilityIndex(rows, "global", "mechanism")
        self.assertEqual(g.candidates({"mechanism": "unknown"}), ([0, 1, 2], True))

    def test_relabel_matches_test_time_rule(self):
        rng = np.random.default_rng(0)
        emb = rng.normal(size=(30, 8)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        centers = rng.normal(size=(4, 8)).astype(np.float32)
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        meta = {0: {"behavior_bucket_key": ["a"]}, 1: {"behavior_bucket_key": ["a"]},
                2: {"behavior_bucket_key": ["b"]}, 3: {"behavior_bucket_key": ["b"]}}
        mems = [{"mechanism": "a" if i % 2 else "b"} for i in range(30)]
        labels = relabel_by_cosine(mems, emb, centers, meta, "constrained", "mechanism")
        idx = CompatibilityIndex([meta[i] for i in range(4)], "constrained", "mechanism")
        for i, m in enumerate(mems):
            cand, _ = idx.candidates(m)
            self.assertEqual(labels[i], nearest_compatible_state(emb[i] @ centers.T, cand, -1, -1).best_index)


class WindowTest(unittest.TestCase):
    def test_last_window_overlaps(self):
        valid = [(i, {"item_id": str(i)}) for i in range(8)]
        wins = chunk_windows("u", valid, 3)
        self.assertEqual([w.positions for w in wins], [[0, 1, 2], [3, 4, 5], [5, 6, 7]])


def _toy_tree(collapse: bool):
    # memories: 4 users, each A,B,B,C with all assignments accepted
    mems, idx_to_state = [], {}
    for u in range(4):
        for t, s in enumerate(["A", "B", "B", "C"]):
            idx_to_state[len(mems)] = s
            mems.append({"user_id": f"u{u}", "window_index": t, "_row_idx": len(mems),
                         "memory_id": f"m{len(mems)}"})
    seqs = build_user_sequences(mems, idx_to_state, collapse_consecutive=collapse)
    stats = accumulate_contexts(seqs, max_order=3, max_evidence=2, seed=0)
    rows = [{"state_id": s, "canonical_text": s, "mechanism": "m", "scope": "s", "direction": "d"}
            for s in "ABC"]
    recs = build_context_records(stats, rows, "user_normalized", 5.0, 2, 2, 0.05, True, 0)
    tree = build_reverse_tree(recs)
    validate_tree_structure(tree, recs)
    return tree


class TreeTest(unittest.TestCase):
    def test_suffix_respects_rejections_and_collapse(self):
        maps = [{"state_id": "A", "assignment_accepted": True},
                {"state_id": "X", "assignment_accepted": False},
                {"state_id": "B", "assignment_accepted": True},
                {"state_id": "B", "assignment_accepted": True}]
        self.assertEqual(latest_accepted_state_suffix(maps), ["B", "B"])
        self.assertEqual(latest_accepted_state_suffix(maps, collapse_consecutive=True), ["B"])

    def test_tree_counts(self):
        tree = _toy_tree(collapse=False)
        node_b = tree["nodes"][tree["nodes"]["ROOT"]["children"]["B"]]
        self.assertEqual(node_b["occurrence_next_counts"], {"B": 4, "C": 4})
        tree_c = _toy_tree(collapse=True)
        node_b = tree_c["nodes"][tree_c["nodes"]["ROOT"]["children"]["B"]]
        self.assertEqual(node_b["occurrence_next_counts"], {"C": 4})  # no self-transitions


class RankingAndMetricsTest(unittest.TestCase):
    def test_sanitize_repairs(self):
        ids = [f"i{k}" for k in range(5)]
        ranking, complete, n, ok, reason, _ = sanitize_ranking(
            json.dumps({"ranking": ["C03", "C03", "C01"]}), ids)
        self.assertEqual(ranking, ["i2", "i0", "i1", "i3", "i4"])
        self.assertFalse(complete)
        self.assertTrue(ok)
        self.assertIn("duplicate", reason)
        ranking, *_ = sanitize_ranking("garbage", ids)
        self.assertEqual(ranking, ids)

    def test_metrics(self):
        m = metrics_from_ranks([1, 2, 11])
        self.assertAlmostEqual(m["Hit@1"], 1 / 3)
        self.assertAlmostEqual(m["NDCG@10"], (1 + 1 / math.log2(3)) / 3)
        self.assertAlmostEqual(m["MRR"], (1 + 0.5 + 1 / 11) / 3)

    def test_paired_bootstrap_direction(self):
        rows = [{"rankings": {"a": {"target_rank": 1}, "b": {"target_rank": 10}}} for _ in range(30)]
        res = paired_bootstrap(rows, "a", "b", n_boot=200)
        self.assertGreater(res["MRR"]["mean_diff"], 0)
        self.assertLess(res["MRR"]["p_two_sided"], 0.05)


class ConfigTest(unittest.TestCase):
    def test_dataset_configs_parse(self):
        import yaml
        root = Path(__file__).resolve().parents[1] / "configs" / "datasets"
        for p in root.glob("*.yaml"):
            cfg = yaml.safe_load(p.read_text())
            for key in ("dataset", "history_size", "tree_dir"):
                self.assertIn(key, cfg, p.name)


class ModePromptTest(unittest.TestCase):
    """Section 2 (WHAT) / section 3 (HOW) of every ranking condition."""

    def _item(self):
        c = [{"item_id": f"i{k}", "title": f"T{k}", "category": "Pop"} for k in range(4)]
        latest = {"recommendation_behavior": "Prefers jazz albums", "semantic_focus": ["jazz"]}
        how = lambda tag: {"mechanism": "deepening", "scope": "same creator",  # noqa: E731
                           "trajectory_signature": f"how-{tag}"}
        return {"history_items": c[:2], "candidate_items": c, "latest_behavior": latest,
                "recent3_behaviors": [latest, latest],
                "hows": {k: how(k) for k in ("latest_how", "tree", "random_how",
                                             "global_continuation", "pmn", "oracle")}}

    def test_sections(self):
        from bt4rec.inference.constants import AB_MODES
        from bt4rec.ranking.modes import mode_prompt, requested_modes
        item = self._item()
        expect = {  # mode -> (WHAT shown, HOW tag shown)
            "native": (False, None), "latest_what_only": (True, None),
            "latest_how_only": (False, "latest_how"), "tree": (True, "tree"), "tree_v2": (True, "tree"),
            "random_how_only": (False, "random_how"), "global_continuation": (True, "global_continuation"),
            "pmn": (True, "pmn"), "oracle": (True, "oracle"),
        }
        self.assertEqual(set(expect), set(AB_MODES))
        for mode, (what, how) in expect.items():
            p = mode_prompt(mode, item)
            self.assertEqual("Prefers jazz albums" in p, what, mode)
            for tag in item["hows"]:
                self.assertEqual(f"how-{tag}" in p, tag == how, (mode, tag))
            self.assertEqual("RECENT OBSERVED BEHAVIORS" in p, mode == "tree_v2", mode)
            self.assertEqual(p.count("OUTPUT REQUIREMENTS"), 1)
        with self.assertRaises(ValueError):
            requested_modes("tree,tree_stay")


class CostTest(unittest.TestCase):
    def test_per_user_and_shared(self):
        from bt4rec.eval.cost import CostTracker
        t = CostTracker("x")
        t.llm("profile", "u1", 100, 10)
        t.llm("profile", "u2", 50, 5)
        t.llm("knowledge", None, 30, 3)
        t.llm("merge", None, 7, 1, phase="offline")
        s = t.summary(["u1", "u2"])
        self.assertEqual(s["online"]["llm_calls_total"], 3)
        self.assertAlmostEqual(s["online"]["llm_calls_per_user"], 1.5)
        self.assertAlmostEqual(s["online"]["input_tokens_per_user"], 90.0)
        self.assertEqual(s["online"]["shared_calls"]["calls"], 1)
        self.assertEqual(s["online"]["attributable_calls_per_user"]["mean"], 1.0)
        self.assertAlmostEqual(s["end_to_end_per_eval_user"]["llm_calls"], 2.0)


if __name__ == "__main__":
    unittest.main()
