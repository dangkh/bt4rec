from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SASRec(nn.Module):
    """Compact SASRec backbone. Padding item id is 0."""

    def __init__(
        self,
        num_items: int,
        maxlen: int = 10,
        hidden_size: int = 50,
        num_blocks: int = 2,
        num_heads: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_items = num_items
        self.maxlen = maxlen
        self.hidden_size = hidden_size
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(maxlen, hidden_size)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_blocks)
        self.final_norm = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.item_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L], right-aligned or full length <= maxlen.
        if seq.ndim != 2:
            raise ValueError("seq must be [B,L]")
        b, l = seq.shape
        if l > self.maxlen:
            seq = seq[:, -self.maxlen :]
            l = self.maxlen
        pos = torch.arange(l, device=seq.device).unsqueeze(0).expand(b, l)
        x = self.item_emb(seq) * math.sqrt(self.hidden_size)
        x = self.dropout(x + self.pos_emb(pos))
        causal = torch.triu(torch.ones(l, l, device=seq.device, dtype=torch.bool), diagonal=1)
        key_padding = seq.eq(0)
        x = self.encoder(x, mask=causal, src_key_padding_mask=key_padding)
        return self.final_norm(x)

    def last_hidden(self, seq: torch.Tensor) -> torch.Tensor:
        h = self.encode(seq)
        mask = seq.ne(0)
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0).expand_as(seq)
        idx = positions.masked_fill(~mask, -1).max(dim=1).values.clamp(min=0)
        return h[torch.arange(seq.size(0), device=seq.device), idx]

    def score_candidates(self, seq: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        user = self.last_hidden(seq)
        item = self.item_emb(candidates)
        return torch.einsum("bd,bkd->bk", user, item)


@dataclass
class SASRecTrainExample:
    seq: list[int]
    pos: list[int]
    neg: list[int]


class SASRecTrainDataset(Dataset):
    def __init__(self, sequences: list[list[int]], num_items: int, maxlen: int, seed: int = 42):
        self.sequences = sequences
        self.num_items = num_items
        self.maxlen = maxlen
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def _negative(self, forbidden: set[int], rng: random.Random) -> int:
        while True:
            x = rng.randint(1, self.num_items)
            if x not in forbidden:
                return x

    def __getitem__(self, idx: int):
        full = self.sequences[idx][-self.maxlen :]
        seq = [0] * self.maxlen
        pos = [0] * self.maxlen
        neg = [0] * self.maxlen
        rng = random.Random(self.seed + self.epoch * 1000003 + idx)
        forbidden = set(full)
        # Predict x[t+1] from x[:t+1].
        for t in range(len(full) - 1):
            p = self.maxlen - len(full) + t
            seq[p] = full[t]
            pos[p] = full[t + 1]
            neg[p] = self._negative(forbidden, rng)
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(pos, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def sasrec_bce_loss(model: SASRec, seq: torch.Tensor, pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    h = model.encode(seq)
    pos_e = model.item_emb(pos)
    neg_e = model.item_emb(neg)
    pos_logits = (h * pos_e).sum(-1)
    neg_logits = (h * neg_e).sum(-1)
    mask = pos.ne(0)
    if not mask.any():
        return h.sum() * 0.0
    pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_logits[mask], torch.ones_like(pos_logits[mask]))
    neg_loss = nn.functional.binary_cross_entropy_with_logits(neg_logits[mask], torch.zeros_like(neg_logits[mask]))
    return pos_loss + neg_loss
