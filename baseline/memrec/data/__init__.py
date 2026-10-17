"""Data loading and sampling utilities"""
from .dataset_frozen import FrozenRecDataset
try:  # legacy loaders need pandas / torch
    from .dataset_base import RecDataset
    from .samplers import (
        BPRSampler,
        SequenceSampler,
        get_bpr_dataloader,
        get_sequence_dataloader
    )
except ImportError:  # pragma: no cover
    RecDataset = None

__all__ = ['FrozenRecDataset', 'RecDataset']
