"""Cache accounting: bytes and attention-work proxies for a kept set."""

from __future__ import annotations

import numpy as np

BYTES_PER_ELEMENT = 2  # fp16, matching kvpress-style accounting


def cache_bytes(keep_mask: np.ndarray, d_k: int, d_v: int) -> int:
    """Bytes occupied by the kept KV pairs across all heads."""
    kept = int(keep_mask.sum())
    return kept * (d_k + d_v) * BYTES_PER_ELEMENT


def keep_fraction(keep_mask: np.ndarray) -> float:
    """Fraction of (head, token) pairs retained."""
    return float(keep_mask.mean())


def attention_flops(n_queries: int, kept_per_head: np.ndarray, d_k: int) -> int:
    """Proxy for attention score FLOPs: queries x kept keys x d_k."""
    return int(n_queries * int(kept_per_head.sum()) * d_k)
