from __future__ import annotations

import argparse


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True)
    p.add_argument("--items", required=True, help="item.json / items.json")
    p.add_argument("--user_sequence", required=True)
    p.add_argument("--user_negative", required=True)
    p.add_argument("--test_behavior", required=True)
    p.add_argument(
        "--candidate_file",
        required=True,
        help="Frozen candidate file from CoMemTree; candidate order is used exactly as stored.",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pretrained_dir", default=None, help="Existing sasrec.pt and stage1.pt directory; defaults to output_dir.")
    p.add_argument("--num_users", type=int, default=300)
    p.add_argument("--history_size", type=int, default=10)
    p.add_argument("--num_negatives", type=int, default=19)
    p.add_argument("--candidate_size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--strict_protocol", action=argparse.BooleanOptionalAction, default=True)

    # SASRec: original A-LLMRec defaults are 50 hidden, 2 blocks, 1 head, 200 epochs.
    p.add_argument("--sasrec_hidden", type=int, default=50)
    p.add_argument("--sasrec_blocks", type=int, default=2)
    p.add_argument("--sasrec_heads", type=int, default=1)
    p.add_argument("--sasrec_dropout", type=float, default=0.2)
    p.add_argument("--sasrec_epochs", type=int, default=200)
    p.add_argument("--sasrec_batch_size", type=int, default=128)
    p.add_argument("--sasrec_lr", type=float, default=1e-3)

    # Stage 1.
    p.add_argument("--text_encoder", default="sentence-transformers/nq-distilbert-base-v1")
    p.add_argument("--train_text_encoder", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--alignment_dim", type=int, default=128)
    p.add_argument("--stage1_epochs", type=int, default=10)
    p.add_argument("--stage1_batch_size", type=int, default=32)
    p.add_argument("--stage1_lr", type=float, default=1e-4)

    # Qwen Stage 2 / evaluation.
    p.add_argument("--model_name", default="unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit")
    p.add_argument("--llm_backend", choices=["unsloth", "transformers"], default="unsloth")
    p.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_seq_length", type=int, default=8192)
    p.add_argument("--max_item_text_chars", type=int, default=160)
    p.add_argument("--stage2_epochs", type=int, default=5)
    p.add_argument("--stage2_batch_size", type=int, default=2)
    p.add_argument("--stage2_lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Batched full-list generation at evaluation time.
    p.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Number of users per Qwen generate() call during evaluation.",
    )
    p.add_argument(
        "--eval_max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens for each generated full ranking.",
    )
    p.add_argument(
        "--eval_debug_outputs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print raw Qwen output and parse result for every evaluated user.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A-LLMRec adapted to CoMemTree: Qwen2.5-7B, configurable user count, up to the last 10 train interactions/user, frozen 20 candidates."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for cmd in ("validate", "train-sasrec", "train-stage1", "train-stage2", "evaluate", "evaluate-native", "all"):
        q = sub.add_parser(cmd)
        add_common(q)
    return parser


def load_data(args):
    if args.candidate_size != args.num_negatives + 1:
        raise ValueError("candidate_size must equal num_negatives + 1")
    from allmrec.data import load_protocol, save_protocol_snapshot
    from allmrec.utils import ensure_dir
    data = load_protocol(
        dataset=args.dataset,
        items_path=args.items,
        sequence_path=args.user_sequence,
        negative_path=args.user_negative,
        test_behavior_path=args.test_behavior,
        candidate_path=args.candidate_file,
        num_users=args.num_users,
        history_size=args.history_size,
        num_negatives=args.num_negatives,
        seed=args.seed,
        strict=args.strict_protocol,
    )
    out = ensure_dir(args.output_dir)
    save_protocol_snapshot(data, str(out / "protocol_snapshot.json"))
    print(f"Dataset={data.dataset} users={len(data.users)} items={data.num_items} fingerprint={data.fingerprint[:12]}")
    lens = [len(u.train) for u in data.users]
    print(f"Protocol: {args.num_users} users, max_history={args.history_size}; "
          f"observed history min/mean/max={min(lens)}/{sum(lens)/len(lens):.2f}/{max(lens)}; "
          f"{args.candidate_size} test candidates")

    # Candidate order must come exactly from the frozen CoMemTree candidate file.
    from collections import Counter

    gt_pos = Counter(
        u.candidates_raw.index(u.gt_raw) + 1
        for u in data.users
    )
    print("GT position distribution:", dict(sorted(gt_pos.items())))
    return data


def main():
    args = build_parser().parse_args()
    if args.llm_backend == "unsloth" and args.command in {"train-stage2", "evaluate", "evaluate-native", "all"}:
        import unsloth  # Must precede torch/transformers imports.
    from allmrec.train import evaluate, evaluate_native, train_sasrec, train_stage1, train_stage2
    data = load_data(args)
    if args.command == "validate":
        print("Protocol validation OK. No model training performed.")
    elif args.command == "train-sasrec":
        train_sasrec(data, args)
    elif args.command == "train-stage1":
        train_stage1(data, args)
    elif args.command == "train-stage2":
        train_stage2(data, args)
    elif args.command == "evaluate":
        evaluate(data, args)
    elif args.command == "evaluate-native":
        evaluate_native(data, args)
    elif args.command == "all":
        train_sasrec(data, args)
        train_stage1(data, args)
        train_stage2(data, args)
        evaluate(data, args)


if __name__ == "__main__":
    main()
