"""Baseline KV-cache compression policies.

Independent re-implementations of the *scoring rules* used by presses in
NVIDIA/kvpress (Apache-2.0), written from published descriptions and
from the kvpress README's own one-line summaries. No kvpress code is
copied or imported. Each policy maps (context, keep_fraction) to a
per-head boolean keep mask with exactly ``round(keep_fraction * n)``
kept positions per head (chunked is the documented exception: it keeps
one representative per block, so its realised keep fraction is
reported, not forced).

Policies never see eval queries. Policies that score from attention use
only the observation-window queries, matching the query-visible
prefill setting kvpress presses assume.
"""

from __future__ import annotations

import numpy as np

from phasekv.attention import AttentionContext


def _exact_topk(scores: np.ndarray, keep: int, protected: np.ndarray) -> np.ndarray:
    """Keep ``protected`` plus the highest-scored unprotected positions."""
    mask = np.zeros(len(scores), dtype=bool)
    mask[protected] = True
    remaining = max(0, keep - int(protected.sum()))
    if remaining == 0:
        return mask
    candidates = np.where(~mask)[0]
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    mask[order[:remaining]] = True
    return mask


def _per_head(context: AttentionContext, keep_fraction: float, scorer) -> np.ndarray:
    n = context.n_tokens
    keep = max(1, int(round(keep_fraction * n)))
    masks = np.zeros((context.n_heads, n), dtype=bool)
    obs_attn = context.attention(context.obs_queries)  # (H, Q, N)
    for h in range(context.n_heads):
        scores = scorer(h, obs_attn[h], context)
        masks[h] = _exact_topk(scores, keep, context.sink_mask)
    return masks


def random_policy(context: AttentionContext, keep_fraction: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = context.n_tokens
    keep = max(1, int(round(keep_fraction * n)))
    masks = np.zeros((context.n_heads, n), dtype=bool)
    for h in range(context.n_heads):
        scores = rng.random(n)
        masks[h] = _exact_topk(scores, keep, context.sink_mask)
    return masks


def streaming_policy(context: AttentionContext, keep_fraction: float) -> np.ndarray:
    """StreamingLLM rule: sinks + most recent window, nothing else."""
    n = context.n_tokens
    keep = max(1, int(round(keep_fraction * n)))
    masks = np.zeros((context.n_heads, n), dtype=bool)
    scores = np.arange(n, dtype=float)  # recency = position
    for h in range(context.n_heads):
        masks[h] = _exact_topk(scores, keep, context.sink_mask)
    return masks


def snapkv_policy(context: AttentionContext, keep_fraction: float) -> np.ndarray:
    """SnapKV rule: mean attention from the observation window queries."""

    def scorer(h: int, attn: np.ndarray, _ctx: AttentionContext) -> np.ndarray:
        return attn.mean(axis=0)

    return _per_head(context, keep_fraction, scorer)


def knorm_policy(context: AttentionContext, keep_fraction: float) -> np.ndarray:
    """Knorm rule: keep low key-norm pairs (inverse norm scoring)."""

    def scorer(h: int, _attn: np.ndarray, ctx: AttentionContext) -> np.ndarray:
        norms = np.linalg.norm(ctx.keys[h], axis=-1)
        return 1.0 / np.maximum(norms, 1e-12)

    return _per_head(context, keep_fraction, scorer)


def expected_attention_policy(context: AttentionContext, keep_fraction: float) -> np.ndarray:
    """Expected-attention rule: score by attention from the mean query."""

    def scorer(h: int, _attn: np.ndarray, ctx: AttentionContext) -> np.ndarray:
        mean_q = ctx.obs_queries[h].mean(axis=0, keepdims=True)
        logits = (ctx.keys[h] @ mean_q[0]) / float(np.sqrt(ctx.d_k))
        logits = logits - logits.max()
        weights = np.exp(logits)
        return weights / weights.sum()

    return _per_head(context, keep_fraction, scorer)


def chunked_policy(
    context: AttentionContext,
    keep_fraction: float,
    block: int | None = None,
) -> np.ndarray:
    """Chunked pooling baseline: one representative per block.

    Each non-overlapping block of ``block`` tokens contributes its
    highest observation-attention token. This is the design whose
    phase sensitivity Zhu et al. document: a needle's fate depends on
    which block it lands in and on its position inside the block.
    ``block`` defaults to the integer whose reciprocal is closest to
    ``keep_fraction`` (2, 4, or 8 for the grid's keep fractions).
    """
    n = context.n_tokens
    if block is None:
        block = max(1, int(round(1.0 / keep_fraction)))
    obs_attn = context.attention(context.obs_queries).mean(axis=1)  # (H, N)
    masks = np.zeros((context.n_heads, n), dtype=bool)
    for h in range(context.n_heads):
        scores = obs_attn[h]
        for start in range(0, n, block):
            stop = min(n, start + block)
            block_idx = np.arange(start, stop)
            # Sinks are always kept; a block containing a sink keeps
            # the sink plus its best-scored member.
            best = block_idx[np.argmax(scores[block_idx])]
            masks[h, best] = True
            masks[h, block_idx[context.sink_mask[block_idx]]] = True
    return masks


def tova_policy(context: AttentionContext, keep_fraction: float) -> np.ndarray:
    """TOVA rule: attention of the last observation query, per head."""

    def scorer(h: int, attn: np.ndarray, _ctx: AttentionContext) -> np.ndarray:
        return attn[-1]

    return _per_head(context, keep_fraction, scorer)


POLICIES = {
    "random": random_policy,
    "streaming": streaming_policy,
    "snapkv": snapkv_policy,
    "knorm": knorm_policy,
    "expected_attention": expected_attention_policy,
    "chunked": chunked_policy,
    "tova": tova_policy,
}


def apply_policy(name: str, context: AttentionContext, keep_fraction: float, seed: int = 0) -> np.ndarray:
    """Apply a named baseline policy. ``random`` consumes ``seed``."""
    if name == "random":
        return random_policy(context, keep_fraction, seed=seed)
    if name not in POLICIES:
        raise ValueError(f"unknown policy: {name}")
    return POLICIES[name](context, keep_fraction)
