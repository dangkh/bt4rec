from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class TwoLayerAligner(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 128, activation: str = "sigmoid"):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.activation = nn.Sigmoid() if activation == "sigmoid" else nn.GELU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.activation(self.encoder(x))
        recon = self.decoder(z)
        return z, recon


@dataclass
class LastTransition:
    user_index: int
    history: list[int]
    target: int
    negative: int


class LastTransitionDataset(Dataset):
    """One final within-train transition per user: first 9 -> 10th for history_size=10."""

    def __init__(self, sequences: list[list[int]], num_items: int, seed: int = 42):
        self.sequences = sequences
        self.num_items = num_items
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        full = self.sequences[idx]
        if len(full) < 2:
            raise ValueError("Need at least 2 interactions for Stage 1/2")
        history, target = full[:-1], full[-1]
        forbidden = set(full)
        rng = random.Random(self.seed + self.epoch * 1000003 + idx)
        while True:
            neg = rng.randint(1, self.num_items)
            if neg not in forbidden:
                break
        return idx, torch.tensor(history, dtype=torch.long), target, neg


def collate_last_transition(batch):
    idxs, histories, targets, negs = zip(*batch)
    maxlen = max(x.numel() for x in histories)
    seq = torch.zeros(len(batch), maxlen, dtype=torch.long)
    for i, x in enumerate(histories):
        seq[i, -x.numel() :] = x
    return (
        torch.tensor(idxs, dtype=torch.long),
        seq,
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(negs, dtype=torch.long),
    )


class Stage1Alignment(nn.Module):
    def __init__(self, rec_dim: int, text_dim: int, latent_dim: int = 128):
        super().__init__()
        self.item_mlp = TwoLayerAligner(rec_dim, latent_dim, "sigmoid")
        self.text_mlp = TwoLayerAligner(text_dim, latent_dim, "sigmoid")

    def loss(
        self,
        user_log: torch.Tensor,
        pos_item: torch.Tensor,
        neg_item: torch.Tensor,
        pos_text: torch.Tensor,
        neg_text: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pos_z, pos_rec = self.item_mlp(pos_item)
        neg_z, neg_rec = self.item_mlp(neg_item)
        pos_tz, pos_trec = self.text_mlp(pos_text)
        neg_tz, neg_trec = self.text_mlp(neg_text)

        pos_logits = (user_log * pos_rec).mean(dim=-1)
        neg_logits = (user_log * neg_rec).mean(dim=-1)
        rec_loss = nn.functional.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
        rec_loss = rec_loss + nn.functional.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        match = nn.functional.mse_loss(pos_z, pos_tz) + nn.functional.mse_loss(neg_z, neg_tz)
        item_recon = nn.functional.mse_loss(pos_rec, pos_item) + nn.functional.mse_loss(neg_rec, neg_item)
        # Detach target text embeddings in reconstruction target, as in the original implementation behavior.
        text_recon = nn.functional.mse_loss(pos_trec, pos_text.detach()) + nn.functional.mse_loss(neg_trec, neg_text.detach())
        total = rec_loss + match + 0.5 * item_recon + 0.2 * text_recon
        return total, {
            "rec": float(rec_loss.detach().cpu()),
            "match": float(match.detach().cpu()),
            "item_recon": float(item_recon.detach().cpu()),
            "text_recon": float(text_recon.detach().cpu()),
        }
