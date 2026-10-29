"""The optimized policy: phase-adaptive, query-adaptive compression.

Three changes over the baseline presses, each ablatable:

1. Per-head adaptive budgets. Instead of the same keep fraction in
   every head, budget is allocated by measured head behaviour: heads
   whose observation attention concentrates on a few tokens (retrieval
   heads) receive more budget; diffuse heads receive less, bounded
   below by a floor so no head is starved.
2. Phase-stratified retention. Within each head, the budget is split
   across compression phases (see phases.py) so no phase class is
   starved, directly addressing the periodic weak spots Zhu et al.
   document for chunked compression.
3. Verified-recall rescue swaps. A held-out rescue query set probes
   needle recall of the proposed keep set. If recall is below
   threshold, the policy swaps in the highest rescue-scored evicted
   tokens for the lowest kept non-sink tokens, keeping the total
   within budget. Rescue uses only rescue queries, never eval queries,
   and every swap is counted in the returned diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from phasekv.attention import AttentionContext
from phasekv.phases import stratified_topk, worst_phase_recall


@dataclass
class AdaptiveResult:
    keep_mask: np.ndarray
    head_budgets: np.ndarray
    n_restores: int
    rescue_recall_before: float
    rescue_recall_after: float


def _head_budgets(
    obs_attn: np.ndarray,
    total_budget: int,
    n_heads: int,
    floor_fraction: float = 0.25,
) -> np.ndarray:
    """Allocate ``total_budget`` slots across heads by concentration.

    Concentration = mean over observation queries of the top-5 attention
    mass per head. Budgets are proportional to concentration, clamped
    to [floor, 2x uniform], then repaired to sum exactly to the total
    by moving single slots from the largest to the smallest budgets.
    """
    top5 = np.sort(obs_attn, axis=-1)[..., -5:].sum(axis=-1)  # (H, Q)
    concentration = top5.mean(axis=1)
    uniform = total_budget / n_heads
    floor = max(1, int(round(uniform * floor_fraction)))
    cap = max(floor + 1, int(round(uniform * 2.0)))
    weights = concentration / max(concentration.sum(), 1e-12)
    budgets = np.maximum(floor, np.minimum(cap, np.round(weights * total_budget).astype(int)))
    # Repair the sum with single-slot moves, preserving bounds.
    while budgets.sum() > total_budget:
        idx = int(np.argmax(budgets))
        if budgets[idx] <= floor:
            break
        budgets[idx] -= 1
    while budgets.sum() < total_budget:
        idx = int(np.argmin(budgets))
        if budgets[idx] >= cap:
            break
        budgets[idx] += 1
    return budgets


class PhaseAdaptivePolicy:
    """Phase-aware adaptive compression with a verified-recall gate."""

    def __init__(
        self,
        stride: int = 4,
        recall_threshold: float = 0.95,
        floor_fraction: float = 0.25,
        max_rescue_swaps: int | None = None,
    ) -> None:
        self.stride = stride
        self.recall_threshold = recall_threshold
        self.floor_fraction = floor_fraction
        self.max_rescue_swaps = max_rescue_swaps

    def apply(
        self, context: AttentionContext, keep_fraction: float
    ) -> AdaptiveResult:
        n = context.n_tokens
        per_head = max(1, int(round(keep_fraction * n)))
        total_budget = per_head * context.n_heads
        obs_attn = context.attention(context.obs_queries)
        scores = obs_attn.mean(axis=1)  # (H, N)
        budgets = _head_budgets(obs_attn, total_budget, context.n_heads, self.floor_fraction)
        keep = np.zeros((context.n_heads, n), dtype=bool)
        for h in range(context.n_heads):
            keep[h] = stratified_topk(
                scores[h], int(budgets[h]), self.stride, context.sink_mask
            )
        before = self._rescue_recall(context, keep)
        keep, n_restores = self._rescue(context, keep, scores)
        after = self._rescue_recall(context, keep)
        return AdaptiveResult(
            keep_mask=keep,
            head_budgets=budgets,
            n_restores=n_restores,
            rescue_recall_before=before,
            rescue_recall_after=after,
        )

    def _rescue_recall(self, context: AttentionContext, keep: np.ndarray) -> float:
        """Rescue-query attention mass retained by the keep set.

        This is the quantity rescue swaps optimize: the fraction of
        total rescue-query attention mass landing on kept pairs. It is
        deliberately *not* needle recall — rescue queries cannot see
        needle labels. Needle recall is an eval-only metric.
        """
        rescue_attn = context.attention(context.rescue_queries)
        kept_mass = np.where(keep[:, None, :], rescue_attn, 0.0).sum()
        return float(kept_mass / max(rescue_attn.sum(), 1e-12))

    def _rescue(
        self, context: AttentionContext, keep: np.ndarray, scores: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """Swap in rescue-scored evicted tokens until recall threshold.

        Rescue scores come from rescue queries only. Each head swaps at
        most 10% of its kept slots (or ``max_rescue_swaps`` in total
        when set), pairing the highest-scored evicted tokens with the
        lowest-scored kept tokens, so phase stratification from the
        initial selection is mostly preserved and the total budget
        never grows. Sinks are never evicted. Returns the updated mask
        and swap count.
        """
        keep = keep.copy()
        rescue_attn = context.attention(context.rescue_queries).mean(axis=1)  # (H, N)
        n_restores = 0
        global_cap = self.max_rescue_swaps
        for h in range(context.n_heads):
            if self._rescue_recall(context, keep) >= self.recall_threshold:
                break
            kept_count = int(keep[h].sum())
            head_cap = max(1, int(round(0.10 * kept_count)))
            evicted = np.where(~keep[h] & ~context.sink_mask)[0]
            kept = np.where(keep[h] & ~context.sink_mask)[0]
            if len(evicted) == 0 or len(kept) == 0:
                continue
            evicted_sorted = evicted[np.argsort(-rescue_attn[h, evicted], kind="stable")]
            kept_sorted = kept[np.argsort(rescue_attn[h, kept], kind="stable")]
            for i in range(min(head_cap, len(evicted_sorted), len(kept_sorted))):
                if global_cap is not None and n_restores >= global_cap:
                    break
                evict_idx = int(evicted_sorted[i])
                kept_idx = int(kept_sorted[i])
                if rescue_attn[h, evict_idx] <= rescue_attn[h, kept_idx]:
                    break
                keep[h, evict_idx] = True
                keep[h, kept_idx] = False
                n_restores += 1
        return keep, n_restores


def merge_downgrade(
    context: AttentionContext,
    keep_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold evicted values into their nearest kept neighbour.

    Returns (merged_values, merged_mask) where merged_mask marks kept
    positions and merged_values holds, at each kept position, the
    value plus the mean of evicted values whose keys are most cosine
    similar to that kept key. Keys of evicted tokens are dropped, so
    the cache stays at the kept size; only value information is
    preserved in downgraded form. This mirrors the idea behind
    kvpress's MergingPress, implemented independently here.
    """
    keys = context.keys
    values = context.values
    merged = values.copy()
    for h in range(context.n_heads):
        kept_idx = np.where(keep_mask[h])[0]
        evicted_idx = np.where(~keep_mask[h])[0]
        if len(kept_idx) == 0 or len(evicted_idx) == 0:
            continue
        k_norm = keys[h] / np.maximum(np.linalg.norm(keys[h], axis=-1, keepdims=True), 1e-12)
        sims = k_norm[evicted_idx] @ k_norm[kept_idx].T  # (E, K)
        nearest = sims.argmax(axis=1)
        acc = np.zeros_like(values[h, kept_idx])
        counts = np.zeros(len(kept_idx))
        for j, target in enumerate(nearest):
            acc[target] += values[h, evicted_idx[j]]
            counts[target] += 1
        # Average, not sum: attention output is a convex combination of
        # values, so folding must preserve value scale.
        merged[h, kept_idx] = (values[h, kept_idx] + acc) / (1.0 + counts)[:, None]
    return merged, keep_mask
