from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import ProtocolData
from .qwen_bridge import QwenBridge, BRIDGE_VERSION
from .metrics import compute_metrics
from .model import ALLMRecQwen, sample_stage2_candidates
from .sasrec import SASRec, SASRecTrainDataset, sasrec_bce_loss
from .stage1 import LastTransitionDataset, Stage1Alignment, TwoLayerAligner, collate_last_transition
from .utils import dump_json, dump_jsonl, ensure_dir, set_seed


def _cost_tracker():
    import sys
    root = Path(__file__).resolve().parents[2] / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from bt4rec.eval.cost import CostTracker, memory_block
    return CostTracker, memory_block


def _write_cost(data: ProtocolData, args, gen_log, rows, method: str, stage: str) -> None:
    """cost_summary.json (bt4rec_cost_v1). online = 1 ranking generation per user;
    offline = Stage-2 training passes of the frozen Qwen (teacher forcing)."""
    CostTracker, memory_block = _cost_tracker()
    tracker = CostTracker(method)
    if len(gen_log) != len(rows):
        raise RuntimeError("generation log does not align with evaluated users")
    for row, (n_in, n_out) in zip(rows, gen_log):
        row["input_tokens"], row["output_tokens"] = n_in, n_out
        tracker.llm(stage, row["user_id"], n_in, n_out)
    extra = {"history_size": args.history_size, "num_users": len(data.users)}
    s2 = Path(args.output_dir) / "stage2_cost.json"
    if stage == "ranking" and s2.exists():
        import json as _json
        rec = _json.loads(s2.read_text(encoding="utf-8"))
        for uid, n_prompt, n_target in rec.get("samples", []):
            tracker.llm("stage2_training_pass", uid, n_prompt, n_target, phase="offline")
        extra["stage2_training"] = {k: v for k, v in rec.items() if k != "samples"}
        extra["offline_note"] = ("Stage-2 passes are forward+backward teacher-forcing steps of the "
                                 "frozen Qwen (input=prompt tokens, output=target-title tokens), "
                                 "not generations. SASRec / Stage-1 use no LLM.")
    params = {}
    for name in ("sasrec.pt", "stage1.pt", "stage2.pt"):
        f = Path(getattr(args, "pretrained_dir", None) or args.output_dir) / name
        if f.exists() or (Path(args.output_dir) / name).exists():
            params[name] = "trained"
    memory = memory_block(0, {u.user_id: 0 for u in data.users},
                          scope="none (parametric: SASRec + aligners + projections)",
                          checkpoints=params)
    tracker.write(Path(args.output_dir) / ("cost_summary.json" if stage == "ranking" else "native_cost_summary.json"),
                  [u.user_id for u in data.users], memory=memory, extra=extra)


def train_sasrec(data: ProtocolData, args) -> Path:
    set_seed(args.seed)
    out = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    model = SASRec(
        data.num_items,
        maxlen=args.history_size,
        hidden_size=args.sasrec_hidden,
        num_blocks=args.sasrec_blocks,
        num_heads=args.sasrec_heads,
        dropout=args.sasrec_dropout,
    ).to(device)
    ds = SASRecTrainDataset([u.train for u in data.users], data.num_items, args.history_size, args.seed)
    loader = DataLoader(ds, batch_size=args.sasrec_batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.sasrec_lr, betas=(0.9, 0.98))
    model.train()
    for epoch in range(1, args.sasrec_epochs + 1):
        ds.set_epoch(epoch)
        losses = []
        for seq, pos, neg in loader:
            seq, pos, neg = seq.to(device), pos.to(device), neg.to(device)
            opt.zero_grad(set_to_none=True)
            loss = sasrec_bce_loss(model, seq, pos, neg)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[SASRec] epoch {epoch}/{args.sasrec_epochs} loss={sum(losses)/max(1,len(losses)):.6f}")
    path = out / "sasrec.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "fingerprint": data.fingerprint,
            "num_items": data.num_items,
            "maxlen": args.history_size,
            "hidden_size": args.sasrec_hidden,
            "num_blocks": args.sasrec_blocks,
            "num_heads": args.sasrec_heads,
            "dropout": args.sasrec_dropout,
        },
        path,
    )
    return path


def load_sasrec(data: ProtocolData, args) -> SASRec:
    ckpt = torch.load(Path(getattr(args, "pretrained_dir", None) or args.output_dir) / "sasrec.pt", map_location="cpu")
    if ckpt.get("fingerprint") != data.fingerprint:
        raise ValueError("SASRec checkpoint protocol fingerprint mismatch")
    model = SASRec(
        ckpt["num_items"], ckpt["maxlen"], ckpt["hidden_size"], ckpt["num_blocks"], ckpt["num_heads"], ckpt["dropout"]
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()
    return model


def _load_text_encoder(name: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("Stage 1 requires sentence-transformers") from e
    model = SentenceTransformer(name, device=device)
    if hasattr(model, "get_sentence_embedding_dimension"):
        dim = int(model.get_sentence_embedding_dimension())
    else:
        dim = int(model.get_embedding_dimension())
    return model, dim


def _encode_texts(model, texts: list[str], device: str) -> torch.Tensor:
    # New sentence-transformers versions deprecate tokenize() in favor of
    # preprocess(), whose returned dict may contain non-Tensor metadata.
    if hasattr(model, "preprocess"):
        features = model.preprocess(texts)
    else:
        features = model.tokenize(texts)
    features = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in features.items()
    }
    outputs = model(features)
    return outputs["sentence_embedding"]


def train_stage1(data: ProtocolData, args) -> Path:
    set_seed(args.seed)
    out = ensure_dir(args.output_dir)
    device = torch.device(args.device)
    sasrec = load_sasrec(data, args)
    for p in sasrec.parameters():
        p.requires_grad = False
    text_model, text_dim = _load_text_encoder(args.text_encoder, args.device)
    if not args.train_text_encoder:
        for p in text_model.parameters():
            p.requires_grad = False
        text_model.eval()
    align = Stage1Alignment(sasrec.hidden_size, text_dim, args.alignment_dim).to(device)

    params = list(align.parameters())
    if args.train_text_encoder:
        params += [p for p in text_model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.stage1_lr, betas=(0.9, 0.98))
    ds = LastTransitionDataset([u.train for u in data.users], data.num_items, args.seed)
    loader = DataLoader(ds, batch_size=args.stage1_batch_size, shuffle=True, collate_fn=collate_last_transition)

    for epoch in range(1, args.stage1_epochs + 1):
        ds.set_epoch(epoch)
        align.train()
        if args.train_text_encoder:
            text_model.train()
        sums = {"total": 0.0, "rec": 0.0, "match": 0.0, "item_recon": 0.0, "text_recon": 0.0}
        count = 0
        for _, history, target, neg in loader:
            history = history.to(device)
            target = target.to(device)
            neg = neg.to(device)
            with torch.no_grad():
                user_log = sasrec.last_hidden(history)
                pos_e = sasrec.item_emb(target)
                neg_e = sasrec.item_emb(neg)
            pos_texts = [data.item_text_raw(data.raw_item(int(x)), args.max_item_text_chars) for x in target.cpu().tolist()]
            neg_texts = [data.item_text_raw(data.raw_item(int(x)), args.max_item_text_chars) for x in neg.cpu().tolist()]
            if args.train_text_encoder:
                pos_t = _encode_texts(text_model, pos_texts, args.device)
                neg_t = _encode_texts(text_model, neg_texts, args.device)
            else:
                with torch.no_grad():
                    pos_t = _encode_texts(text_model, pos_texts, args.device)
                    neg_t = _encode_texts(text_model, neg_texts, args.device)
            opt.zero_grad(set_to_none=True)
            loss, parts = align.loss(user_log, pos_e, neg_e, pos_t, neg_t)
            loss.backward()
            opt.step()
            sums["total"] += float(loss.detach().cpu())
            for k, v in parts.items():
                sums[k] += v
            count += 1
        msg = " ".join(f"{k}={v/max(1,count):.5f}" for k, v in sums.items())
        print(f"[Stage1] epoch {epoch}/{args.stage1_epochs} {msg}")

    path = out / "stage1.pt"
    torch.save(
        {
            "fingerprint": data.fingerprint,
            "rec_dim": sasrec.hidden_size,
            "text_dim": text_dim,
            "alignment_dim": args.alignment_dim,
            "item_mlp": align.item_mlp.state_dict(),
            "text_mlp": align.text_mlp.state_dict(),
            "text_encoder": args.text_encoder,
            "train_text_encoder": args.train_text_encoder,
        },
        path,
    )
    return path


def load_item_aligner(data: ProtocolData, args, sasrec: SASRec) -> TwoLayerAligner:
    ckpt = torch.load(Path(getattr(args, "pretrained_dir", None) or args.output_dir) / "stage1.pt", map_location="cpu")
    if ckpt.get("fingerprint") != data.fingerprint:
        raise ValueError("Stage1 checkpoint protocol fingerprint mismatch")
    m = TwoLayerAligner(sasrec.hidden_size, ckpt["alignment_dim"], "sigmoid")
    m.load_state_dict(ckpt["item_mlp"])
    m.to(args.device).eval()
    return m


def _build_stage2_model(data: ProtocolData, args) -> ALLMRecQwen:
    sasrec = load_sasrec(data, args)
    item_aligner = load_item_aligner(data, args, sasrec)
    bridge = QwenBridge(
        args.model_name,
        args.device,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        backend=args.llm_backend,
    )
    # Do not call `.to()` on the full wrapper: a bitsandbytes 4-bit Qwen
    # is already placed by Unsloth/Transformers and may reject a later `.to()`.
    model = ALLMRecQwen(sasrec, item_aligner, bridge, data, args.max_item_text_chars)
    model.user_proj.to(args.device)
    model.item_proj.to(args.device)
    return model


def train_stage2(data: ProtocolData, args) -> Path:
    set_seed(args.seed)
    out = ensure_dir(args.output_dir)
    model = _build_stage2_model(data, args)
    model.bridge.set_training_mode()
    trainable = list(model.user_proj.parameters()) + list(model.item_proj.parameters())
    opt = torch.optim.Adam(trainable, lr=args.stage2_lr, betas=(0.9, 0.98))
    ds = LastTransitionDataset([u.train for u in data.users], data.num_items, args.seed)
    loader = DataLoader(ds, batch_size=args.stage2_batch_size, shuffle=True, collate_fn=collate_last_transition)
    stage2_samples: list = []

    for epoch in range(1, args.stage2_epochs + 1):
        ds.set_epoch(epoch)
        losses = []
        for user_idx, history, target, _ in loader:
            history = history.to(args.device)
            target = target.to(args.device)
            candidate_lists = []
            for j in range(history.size(0)):
                uid_idx = int(user_idx[j])
                rng = random.Random(args.seed + epoch * 1000003 + uid_idx)
                hist = history[j][history[j] > 0].tolist()
                candidate_lists.append(
                    sample_stage2_candidates(hist, int(target[j]), data.num_items, args.candidate_size, rng)
                )
            opt.zero_grad(set_to_none=True)
            log_start = len(model.bridge.train_log)
            loss = model.stage2_loss(history, target, candidate_lists)
            for j, (n_prompt, n_target) in enumerate(model.bridge.train_log[log_start:]):
                stage2_samples.append([data.users[int(user_idx[j])].user_id, n_prompt, n_target])
            if not loss.requires_grad or not torch.isfinite(loss):
                raise RuntimeError("Stage2 loss has no gradient or is not finite")
            loss.backward()
            for name, projection in (("user", model.user_proj), ("item", model.item_proj)):
                grads = [p.grad for p in projection.parameters() if p.grad is not None]
                if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
                    raise RuntimeError(f"Stage2 {name} projection gradients missing/zero/non-finite; check Unsloth version")
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[Stage2] epoch {epoch}/{args.stage2_epochs} loss={sum(losses)/max(1,len(losses)):.6f}")
        torch.cuda.empty_cache()

    dump_json({"epochs": args.stage2_epochs, "users": len(data.users),
               "training_passes": len(stage2_samples),
               "prompt_tokens_total": sum(x[1] for x in stage2_samples),
               "target_tokens_total": sum(x[2] for x in stage2_samples),
               "samples": stage2_samples}, out / "stage2_cost.json")
    path = out / "stage2.pt"
    torch.save(
        {
            "fingerprint": data.fingerprint,
            "model_name": args.model_name,
            "bridge_version": BRIDGE_VERSION,
            "user_proj": model.user_proj.state_dict(),
            "item_proj": model.item_proj.state_dict(),
            "hidden_size": model.bridge.hidden_size,
        },
        path,
    )
    return path


def load_stage2_model(data: ProtocolData, args) -> ALLMRecQwen:
    ckpt = torch.load(Path(args.output_dir) / "stage2.pt", map_location="cpu")
    if ckpt.get("fingerprint") != data.fingerprint:
        raise ValueError("Stage2 checkpoint protocol fingerprint mismatch")
    if ckpt.get("model_name") != args.model_name:
        raise ValueError(f"Stage2 checkpoint model={ckpt.get('model_name')} but CLI model={args.model_name}")
    if ckpt.get("bridge_version") != BRIDGE_VERSION:
        raise ValueError("Retrain Stage2 with this Qwen chat bridge; old projections are incompatible")
    model = _build_stage2_model(data, args)
    model.user_proj.load_state_dict(ckpt["user_proj"])
    model.item_proj.load_state_dict(ckpt["item_proj"])
    model.eval()
    model.bridge.set_inference_mode()
    return model


def evaluate(data: ProtocolData, args) -> dict[str, float]:
    model = load_stage2_model(data, args)
    ranks = []
    rows = []
    parse_failures = 0
    parse_reasons: dict[str, int] = {}
    identity_rankings = 0
    repaired_rankings = 0

    eval_batch_size = max(1, int(args.eval_batch_size))
    max_new_tokens = int(args.eval_max_new_tokens)

    pbar = tqdm(
        total=len(data.users),
        desc=f"A-LLMRec full-list ranking (batch={eval_batch_size})",
    )

    for start in range(0, len(data.users), eval_batch_size):
        batch_users = data.users[start : start + eval_batch_size]

        batch_results = model.rank_batch(
            histories=[u.train for u in batch_users],
            candidate_lists=[u.candidates for u in batch_users],
            max_new_tokens=max_new_tokens,
            debug=args.eval_debug_outputs,
        )

        for u, result in zip(batch_users, batch_results):
            ranked, raw_output, parse_ok, parse_reason, parsed_order = result
            repaired_rankings += int(parse_reason == "repaired")

            if not parse_ok:
                parse_failures += 1
                parse_reasons[parse_reason] = (
                    parse_reasons.get(parse_reason, 0) + 1
                )
                if args.eval_debug_outputs:
                    print(
                        f"[WARN] ranking parse failed for user={u.user_id}; "
                        f"reason={parse_reason}; falling back to candidate order"
                    )
            elif parsed_order == list(range(1, len(u.candidates) + 1)):
                identity_rankings += 1

            # Safety: evaluation must always rank the exact frozen candidate pool.
            assert len(ranked) == len(u.candidates) == args.candidate_size
            assert set(ranked) == set(u.candidates)
            assert len(set(ranked)) == len(ranked)

            rank = ranked.index(u.gt) + 1
            ranks.append(rank)

            rows.append(
                {
                    "user_id": u.user_id,
                    "gt": u.gt_raw,
                    "rank": rank,
                    "ranked_item_ids": [data.raw_item(x) for x in ranked],
                    "candidate_item_ids_input_order": u.candidates_raw,
                    "raw_output": raw_output,
                    "parse_ok": parse_ok,
                    "parse_reason": parse_reason,
                    "parsed_candidate_order_1based": parsed_order,
                    "fallback_used": not parse_ok,
                }
            )

        pbar.update(len(batch_users))

    pbar.close()

    metrics = compute_metrics(ranks, ks=(1, 3, 5, 10, 15, 20))
    n_users = max(1, len(data.users))
    metrics["ParseSuccessRate"] = (
        len(data.users) - parse_failures
    ) / n_users
    metrics["ParseFailures"] = parse_failures
    metrics["RepairedRankings"] = repaired_rankings
    metrics["CompleteRankingRate"] = (len(data.users) - parse_failures - repaired_rankings) / n_users
    metrics["IdentityRankingRate"] = identity_rankings / n_users
    metrics["IdentityRankings"] = identity_rankings
    metrics["EvalBatchSize"] = eval_batch_size

    out = ensure_dir(args.output_dir)
    _write_cost(data, args, model.bridge.gen_log, rows, "allmrec", "ranking")
    dump_jsonl(rows, out / "rankings.jsonl")
    dump_json(metrics, out / "metrics.json")

    print(f"Parse success: {len(data.users) - parse_failures}/{len(data.users)}")
    print(f"Parse failure: {parse_failures}/{len(data.users)}")
    print(f"Identity rankings: {identity_rankings}/{len(data.users)}")
    if parse_reasons:
        print(
            "Parse failure reasons:",
            json.dumps(parse_reasons, indent=2),
        )
    print(json.dumps(metrics, indent=2))
    return metrics

@torch.no_grad()
def evaluate_native(data: ProtocolData, args) -> dict[str, float]:
    """Text-only frozen-Qwen sanity check.

    This path intentionally does NOT load or use:
      - sasrec.pt,
      - stage1.pt,
      - stage2.pt,
      - SASRec user/item embeddings,
      - Stage-1 aligned item embeddings,
      - Stage-2 learned projections.

    It uses only:
      - the same frozen Qwen backbone,
      - the same observed TRAIN-history titles,
      - the same frozen candidate titles and candidate order,
      - the same C-label output grammar and ranking parser.

    This is a diagnostic baseline, not the full A-LLMRec method.
    """
    set_seed(args.seed)

    print(
        "[Native sanity check] Frozen Qwen + history titles + candidate titles. "
        "No SASRec/Stage1/Stage2 checkpoint is loaded."
    )

    bridge = QwenBridge(
        args.model_name,
        args.device,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        backend=args.llm_backend,
    )

    ranks = []
    rows = []
    parse_failures = 0
    parse_reasons: dict[str, int] = {}
    identity_rankings = 0
    repaired_rankings = 0

    eval_batch_size = max(1, int(args.eval_batch_size))
    max_new_tokens = int(args.eval_max_new_tokens)

    pbar = tqdm(
        total=len(data.users),
        desc=f"Native text-only ranking (batch={eval_batch_size})",
    )

    for start in range(0, len(data.users), eval_batch_size):
        batch_users = data.users[start : start + eval_batch_size]

        prompts = []
        num_candidates = []

        for u in batch_users:
            history_titles = [
                data.title_raw(
                    data.raw_item(int(item_idx)),
                    args.max_item_text_chars,
                )
                for item_idx in u.train
            ]
            candidate_titles = [
                data.title_raw(
                    data.raw_item(int(item_idx)),
                    args.max_item_text_chars,
                )
                for item_idx in u.candidates
            ]

            prompts.append(
                bridge.make_native_ranking_prompt(
                    history_titles,
                    candidate_titles,
                )
            )
            num_candidates.append(len(u.candidates))

        generations = bridge.generate_native_ranking_batch(
            prompts=prompts,
            num_candidates=num_candidates,
            max_new_tokens=max_new_tokens,
            debug=args.eval_debug_outputs,
        )

        for u, gen in zip(batch_users, generations):
            parsed_order = list(gen.order_1based)
            repaired_rankings += int(gen.parse_reason == "repaired")

            if gen.parse_ok:
                ranked = [
                    u.candidates[i - 1]
                    for i in parsed_order
                ]
            else:
                # Same conservative fallback as the full A-LLMRec evaluator:
                # preserve the exact frozen input candidate order.
                ranked = list(u.candidates)
                parse_failures += 1
                parse_reasons[gen.parse_reason] = (
                    parse_reasons.get(gen.parse_reason, 0) + 1
                )
                if args.eval_debug_outputs:
                    print(
                        f"[WARN] native ranking parse failed for user={u.user_id}; "
                        f"reason={gen.parse_reason}; falling back to candidate order"
                    )

            if gen.parse_ok and parsed_order == list(
                range(1, len(u.candidates) + 1)
            ):
                identity_rankings += 1

            # Safety: native evaluation must use exactly the frozen pool.
            assert len(ranked) == len(u.candidates) == args.candidate_size
            assert set(ranked) == set(u.candidates)
            assert len(set(ranked)) == len(ranked)

            rank = ranked.index(u.gt) + 1
            ranks.append(rank)

            rows.append(
                {
                    "mode": "native_text_only",
                    "user_id": u.user_id,
                    "gt": u.gt_raw,
                    "rank": rank,
                    "ranked_item_ids": [
                        data.raw_item(x)
                        for x in ranked
                    ],
                    "candidate_item_ids_input_order": u.candidates_raw,
                    "raw_output": gen.raw_output,
                    "parse_ok": gen.parse_ok,
                    "parse_reason": gen.parse_reason,
                    "parsed_candidate_order_1based": parsed_order,
                    "fallback_used": not gen.parse_ok,
                    "uses_sasrec_embedding": False,
                    "uses_stage1": False,
                    "uses_stage2": False,
                }
            )

        pbar.update(len(batch_users))

    pbar.close()

    metrics = compute_metrics(
        ranks,
        ks=(1, 3, 5, 10, 15, 20),
    )
    n_users = max(1, len(data.users))
    metrics["ParseSuccessRate"] = (
        len(data.users) - parse_failures
    ) / n_users
    metrics["ParseFailures"] = parse_failures
    metrics["RepairedRankings"] = repaired_rankings
    metrics["CompleteRankingRate"] = (len(data.users) - parse_failures - repaired_rankings) / n_users
    metrics["IdentityRankingRate"] = identity_rankings / n_users
    metrics["IdentityRankings"] = identity_rankings
    metrics["EvalBatchSize"] = eval_batch_size
    metrics["UsesSASRecEmbedding"] = 0
    metrics["UsesStage1"] = 0
    metrics["UsesStage2"] = 0

    out = ensure_dir(args.output_dir)
    _write_cost(data, args, bridge.gen_log, rows, "allmrec_native_text_only", "native_ranking")

    # Use separate filenames so native sanity-check results NEVER overwrite
    # the full A-LLMRec evaluation artifacts.
    dump_jsonl(
        rows,
        out / "native_rankings.jsonl",
    )
    dump_json(
        metrics,
        out / "native_metrics.json",
    )

    print(
        f"Native parse success: "
        f"{len(data.users) - parse_failures}/{len(data.users)}"
    )
    print(
        f"Native parse failure: "
        f"{parse_failures}/{len(data.users)}"
    )
    print(
        f"Native identity rankings: "
        f"{identity_rankings}/{len(data.users)}"
    )
    if parse_reasons:
        print(
            "Native parse failure reasons:",
            json.dumps(parse_reasons, indent=2),
        )
    print(json.dumps(metrics, indent=2))
    return metrics

