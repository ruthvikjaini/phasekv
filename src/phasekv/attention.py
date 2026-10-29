"""Simulated multi-head attention contexts for KV-cache experiments.

The testbed models the quantities KV-cache compression actually acts on:
per-head keys, values, and queries, with planted needle facts, distractor
clusters, attention sinks, and per-head specialization (retrieval heads vs
local heads). Nothing here is a real transformer: it is a controlled
simulator in which every compression policy sees exactly the same keys,
values, and queries, so differences in recall are attributable to the
policy, not to model or data drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AttentionContext:
    """One simulated prefill context.

    Attributes:
        keys: (n_heads, n_tokens, d_k) key vectors.
        values: (n_heads, n_tokens, d_v) value vectors.
        obs_queries: (n_heads, n_obs, d_k) queries visible at compression
            time (the observation window scoring may use).
        rescue_queries: (n_heads, n_probe, d_k) held-out queries the
            adaptive policy may use for its recall-probe rescue gate.
        eval_queries: (n_heads, n_probe, d_k) held-out queries used only
            for evaluation, never for scoring or rescue.
        needle_mask: (n_tokens,) True where a token carries a planted fact.
        sink_mask: (n_tokens,) True for attention-sink tokens.
        n_heads, n_tokens, d_k, d_v: shapes.
        seed: the seed that generated this context.
    """

    keys: np.ndarray
    values: np.ndarray
    obs_queries: np.ndarray
    rescue_queries: np.ndarray
    eval_queries: np.ndarray
    needle_mask: np.ndarray
    sink_mask: np.ndarray
    seed: int = 0
    needle_group: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    """Per-token group: 0 = ordinary, 1 = group-A needle, 2 = group-B needle."""

    @property
    def n_heads(self) -> int:
        return self.keys.shape[0]

    @property
    def n_tokens(self) -> int:
        return self.keys.shape[1]

    @property
    def d_k(self) -> int:
        return self.keys.shape[2]

    @property
    def d_v(self) -> int:
        return self.values.shape[2]

    def attention(self, queries: np.ndarray) -> np.ndarray:
        """Softmax attention weights for queries against all keys.

        queries: (n_heads, n_queries, d_k). Returns (n_heads, n_queries,
        n_tokens) rows summing to 1.
        """
        scale = float(np.sqrt(self.d_k))
        logits = np.einsum("hqd,hkd->hqk", queries, self.keys) / scale
        logits = logits - logits.max(axis=-1, keepdims=True)
        weights = np.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True)

    def output(self, queries: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
        """Attention output using only kept KV pairs, renormalized.

        keep_mask: (n_heads, n_tokens) bool. Returns (n_heads, n_queries,
        d_v). Positions with no kept pair produce a zero vector.
        """
        weights = self.attention(queries)
        masked = np.where(keep_mask[:, None, :], weights, 0.0)
        denom = masked.sum(axis=-1, keepdims=True)
        safe = np.where(denom > 0, masked / np.where(denom == 0, 1.0, denom), 0.0)
        return np.einsum("hqk,hkd->hqd", safe, self.values)

    def full_output(self, queries: np.ndarray) -> np.ndarray:
        """Attention output with the full, uncompressed cache."""
        weights = self.attention(queries)
        return np.einsum("hqk,hkd->hqd", weights, self.values)


def _unit(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    vectors = rng.normal(size=shape)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


def make_context(
    seed: int = 0,
    n_tokens: int = 512,
    n_heads: int = 8,
    d_k: int = 32,
    d_v: int = 32,
    n_needles: int = 16,
    needle_strength: float = 3.0,
    n_sinks: int = 4,
    n_distractor_clusters: int = 4,
    n_obs_queries: int = 8,
    n_probe_queries: int = 8,
    retrieval_fraction: float = 0.5,
) -> AttentionContext:
    """Generate one deterministic simulated context.

    Needle tokens carry a per-head direction that scoring queries also
    point at, so a good policy must retain them. Distractor clusters
    share a direction that scoring queries do *not* point at, so a
    policy that keeps only high-norm or high-similarity mass can be
    lured. Retrieval heads (the first ``retrieval_fraction`` of heads)
    have strong needle alignment; local heads align with recent tokens
    instead. Needle positions are spread uniformly so every compression
    phase contains needles.
    """
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(n_heads, n_tokens, d_k))
    values = rng.normal(size=(n_heads, n_tokens, d_v))

    needle_mask = np.zeros(n_tokens, dtype=bool)
    sink_mask = np.zeros(n_tokens, dtype=bool)
    sink_mask[:n_sinks] = True

    # Needle positions: evenly spread, offset so they are not aligned
    # with chunk boundaries.
    needle_positions = np.linspace(n_sinks + 3, n_tokens - 4, n_needles).round().astype(int)
    needle_positions = np.unique(np.clip(needle_positions, n_sinks, n_tokens - 1))
    needle_mask[needle_positions] = True

    needle_dirs = _unit(rng, (n_heads, d_k))
    needle_dirs_b = _unit(rng, (n_heads, d_k))
    n_retrieval = max(1, int(round(n_heads * retrieval_fraction)))
    # Two needle groups, alternating along the sequence so both spread
    # across every compression phase. Group A is what observation
    # queries point at; group B is what later (rescue/eval) queries
    # point at. This is the shared-context reuse setting: compression
    # happens before the question is known, and query-blind scoring
    # keeps group A while losing group B.
    group = np.zeros(n_tokens, dtype=int)
    for i, pos in enumerate(needle_positions):
        group[pos] = 1 if i % 2 == 0 else 2
    for h in range(n_heads):
        strength = needle_strength if h < n_retrieval else needle_strength * 0.35
        a_pos = needle_positions[group[needle_positions] == 1]
        b_pos = needle_positions[group[needle_positions] == 2]
        keys[h, a_pos] = (
            needle_dirs[h] * strength + rng.normal(scale=0.3, size=(len(a_pos), d_k))
        )
        keys[h, b_pos] = (
            needle_dirs_b[h] * strength + rng.normal(scale=0.3, size=(len(b_pos), d_k))
        )

    # Distractor clusters: shared directions, moderate strength, placed
    # between needles.
    cluster_dirs = _unit(rng, (n_distractor_clusters, d_k))
    free = np.where(~needle_mask & ~sink_mask)[0]
    rng.shuffle(free)
    per_cluster = max(1, len(free) // (n_distractor_clusters * 2))
    for c in range(n_distractor_clusters):
        members = free[c * per_cluster : (c + 1) * per_cluster]
        if len(members) == 0:
            continue
        keys[:, members] += cluster_dirs[c] * 1.2

    # Unit-normalise every key. This removes a norm artifact: without
    # it, needle construction makes needle keys systematically
    # low-norm, so the Knorm rule (keep low-norm keys) would keep every
    # needle trivially. With unit-norm keys, discrimination is purely
    # directional, as in attention itself, and Knorm's signal is weak
    # by construction, matching its status as a cheap heuristic.
    keys = keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)

    def _queries(n_q: int, stream: np.random.Generator, which: str) -> np.ndarray:
        queries = np.empty((n_heads, n_q, d_k))
        for h in range(n_heads):
            if h < n_retrieval:
                # Observation queries point at group A; rescue and eval
                # queries point at group B (the later question's facts).
                base = needle_dirs[h] if which == "obs" else needle_dirs_b[h]
            else:
                # Local heads: align with the recent window's mean key.
                base = keys[h, -32:].mean(axis=0)
                base = base / max(np.linalg.norm(base), 1e-12)
            raw = base + stream.normal(scale=0.25, size=(n_q, d_k))
            queries[h] = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
        return queries

    obs_queries = _queries(n_obs_queries, np.random.default_rng(seed + 1_000_001), "obs")
    rescue_queries = _queries(n_probe_queries, np.random.default_rng(seed + 2_000_001), "probe")
    eval_queries = _queries(n_probe_queries, np.random.default_rng(seed + 3_000_001), "probe")
    return AttentionContext(
        keys=keys,
        values=values,
        obs_queries=obs_queries,
        rescue_queries=rescue_queries,
        eval_queries=eval_queries,
        needle_mask=needle_mask,
        sink_mask=sink_mask,
        seed=seed,
        needle_group=group,
    )
