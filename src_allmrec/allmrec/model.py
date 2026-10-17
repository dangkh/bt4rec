from __future__ import annotations

import random
from typing import Sequence

import torch
import torch.nn as nn

from .data import ProtocolData
from .qwen_bridge import QwenBridge
from .sasrec import SASRec
from .stage1 import TwoLayerAligner


class Projection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: str):
        super().__init__()
        act = nn.LeakyReLU() if activation == "leaky_relu" else nn.GELU()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            act,
            nn.Linear(out_dim, out_dim),
        )
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[3].weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ALLMRecQwen(nn.Module):
    def __init__(
        self,
        sasrec: SASRec,
        item_aligner: TwoLayerAligner,
        bridge: QwenBridge,
        data: ProtocolData,
        max_item_text_chars: int = 160,
    ):
        super().__init__()
        self.sasrec = sasrec
        self.item_aligner = item_aligner
        self.bridge = bridge
        self.data = data
        self.max_item_text_chars = max_item_text_chars
        for p in self.sasrec.parameters():
            p.requires_grad = False
        for p in self.item_aligner.parameters():
            p.requires_grad = False
        self.sasrec.eval()
        self.item_aligner.eval()
        self.user_proj = Projection(sasrec.hidden_size, bridge.hidden_size, "leaky_relu")
        self.item_proj = Projection(item_aligner.encoder.out_features, bridge.hidden_size, "gelu")

    def raw_title(self, idx: int) -> str:
        return self.data.title_raw(self.data.raw_item(idx), self.max_item_text_chars)

    @torch.no_grad()
    def user_rep(self, history: torch.Tensor) -> torch.Tensor:
        return self.sasrec.last_hidden(history)

    @torch.no_grad()
    def joint_item(self, item_ids: torch.Tensor) -> torch.Tensor:
        e = self.sasrec.item_emb(item_ids)
        z, _ = self.item_aligner(e)
        return z

    def projected_context(
        self,
        histories: torch.Tensor,
        candidate_lists: Sequence[Sequence[int]],
    ):
        with torch.no_grad():
            u = self.sasrec.last_hidden(histories)
        user_p = self.user_proj(u)
        hist_out = []
        cand_out = []
        for i in range(histories.size(0)):
            hist_ids = histories[i][histories[i] > 0]
            with torch.no_grad():
                hz = self.joint_item(hist_ids)
                cz = self.joint_item(
                    torch.tensor(candidate_lists[i], dtype=torch.long, device=histories.device)
                )
            hist_out.append(self.item_proj(hz))
            cand_out.append(self.item_proj(cz))
        return user_p, hist_out, cand_out

    def stage2_loss(
        self,
        histories: torch.Tensor,
        targets: torch.Tensor,
        candidate_lists: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Keep A-LLMRec's single-positive Stage-2 training objective.

        ``targets`` are the final TRAIN interactions from LastTransitionDataset;
        they are not test GT items.
        """
        user_p, hist_p, cand_p = self.projected_context(histories, candidate_lists)
        prompts = []
        target_titles = []
        for i in range(histories.size(0)):
            hist_ids = histories[i][histories[i] > 0].tolist()
            hist_titles = [self.raw_title(x) for x in hist_ids]
            cand_titles = [self.raw_title(x) for x in candidate_lists[i]]
            prompts.append(self.bridge.make_prompt(hist_titles, cand_titles))
            target_titles.append(self.raw_title(int(targets[i].item())))
        return self.bridge.training_loss(prompts, user_p, hist_p, cand_p, target_titles)

    @torch.no_grad()
    def rank_batch(
        self,
        histories: Sequence[Sequence[int]],
        candidate_lists: Sequence[Sequence[int]],
        max_new_tokens: int = 1024,
        debug: bool = False,
    ):
        """Batch full-list generation for evaluation.

        Each user's candidate list is kept independent.  If that user's generated
        permutation parses successfully, it is used exactly as generated.
        Otherwise only that user falls back to the exact frozen input candidate
        order.
        """
        if len(histories) != len(candidate_lists):
            raise ValueError("histories and candidate_lists must have the same size")
        if len(histories) == 0:
            return []

        device = next(self.user_proj.parameters()).device
        maxlen = int(self.sasrec.maxlen)

        # Match SASRec training convention: sequences are right-aligned and
        # padding item id 0 is on the LEFT.
        history_tensor = torch.zeros(
            (len(histories), maxlen),
            dtype=torch.long,
            device=device,
        )
        normalized_histories: list[list[int]] = []
        normalized_candidates: list[list[int]] = []

        for i, (history, candidates) in enumerate(zip(histories, candidate_lists)):
            hist = list(history)[-maxlen:]
            cands = list(candidates)

            if not hist:
                raise ValueError(f"history is empty for batch index {i}")
            if not cands:
                raise ValueError(f"candidate list is empty for batch index {i}")
            if len(set(cands)) != len(cands):
                raise ValueError(
                    f"candidate list contains duplicate item IDs at batch index {i}"
                )

            history_tensor[i, -len(hist):] = torch.tensor(
                hist,
                dtype=torch.long,
                device=device,
            )
            normalized_histories.append(hist)
            normalized_candidates.append(cands)

        user_p, hist_p, cand_p = self.projected_context(
            history_tensor,
            normalized_candidates,
        )

        prompts = []
        for history, candidates in zip(
            normalized_histories,
            normalized_candidates,
        ):
            hist_titles = [self.raw_title(x) for x in history]
            cand_titles = [self.raw_title(x) for x in candidates]
            prompts.append(
                self.bridge.make_ranking_prompt(hist_titles, cand_titles)
            )

        generations = self.bridge.generate_ranking_batch(
            prompts=prompts,
            user_embs=user_p,
            history_embs=hist_p,
            candidate_embs=cand_p,
            num_candidates=[len(x) for x in normalized_candidates],
            max_new_tokens=max_new_tokens,
            debug=debug,
        )

        outputs = []
        for candidates, gen in zip(normalized_candidates, generations):
            if gen.parse_ok:
                # Keep valid Qwen ranking exactly as generated.
                ranked = [candidates[i - 1] for i in gen.order_1based]
            else:
                # Same conservative policy as CoMemTree: only parse failure
                # falls back to the exact frozen candidate order.
                ranked = list(candidates)

            assert len(ranked) == len(candidates)
            assert len(set(ranked)) == len(candidates)
            assert set(ranked) == set(candidates)

            outputs.append(
                (
                    ranked,
                    gen.raw_output,
                    gen.parse_ok,
                    gen.parse_reason,
                    gen.order_1based,
                )
            )

        return outputs

    @torch.no_grad()
    def rank_one(
        self,
        history: Sequence[int],
        candidates: Sequence[int],
        max_new_tokens: int = 1024,
        debug: bool = False,
    ):
        """Single-user wrapper preserving the previous API."""
        return self.rank_batch(
            histories=[history],
            candidate_lists=[candidates],
            max_new_tokens=max_new_tokens,
            debug=debug,
        )[0]


def sample_stage2_candidates(
    history: Sequence[int], target: int, num_items: int, candidate_size: int, rng: random.Random
) -> list[int]:
    """Stage-2 TRAIN candidate sampling.

    The positive target is a TRAIN interaction.  It is shuffled together with
    sampled negatives so Stage 2 cannot learn a fixed positive position.
    """
    forbidden = set(history) | {target}
    negs = []
    while len(negs) < candidate_size - 1:
        x = rng.randint(1, num_items)
        if x not in forbidden and x not in negs:
            negs.append(x)
    cands = [target] + negs
    rng.shuffle(cands)
    return cands
