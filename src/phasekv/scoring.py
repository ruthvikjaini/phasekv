"""Grid scoring: run every policy over shared contexts and aggregate."""

from __future__ import annotations

import numpy as np

from phasekv.adaptive import PhaseAdaptivePolicy, merge_downgrade
from phasekv.attention import make_context
from phasekv.policies import apply_policy
from phasekv.probes import evaluate

BASELINE_NAMES = [
    "random",
    "streaming",
    "snapkv",
    "knorm",
    "expected_attention",
    "chunked",
    "tova",
]


def score_policy(
    name: str,
    n_tokens: int,
    keep_fraction: float,
    seeds: list[int],
    stride: int = 4,
    merge: bool = False,
) -> dict:
    """Mean metrics for one policy over ``seeds`` shared contexts."""
    rows: list[dict] = []
    restores: list[int] = []
    for seed in seeds:
        context = make_context(seed=seed, n_tokens=n_tokens)
        if name in ("phase_adaptive", "phase_adaptive_norescue"):
            threshold = 0.95 if name == "phase_adaptive" else 0.0
            result = PhaseAdaptivePolicy(stride=stride, recall_threshold=threshold).apply(
                context, keep_fraction
            )
            mask = result.keep_mask
            restores.append(result.n_restores)
            merged = None
            if merge:
                merged, _ = merge_downgrade(context, mask)
            rows.append(evaluate(context, mask, stride=stride, merged_values=merged))
        else:
            mask = apply_policy(name, context, keep_fraction, seed=seed)
            merged = None
            if merge:
                merged, _ = merge_downgrade(context, mask)
            rows.append(evaluate(context, mask, stride=stride, merged_values=merged))
    agg: dict = {
        "policy": name,
        "n_tokens": n_tokens,
        "keep_fraction": keep_fraction,
        "n_seeds": len(seeds),
        "n_restores_mean": float(np.mean(restores)) if restores else 0.0,
    }
    for key in ("needle_recall", "worst_phase_recall", "mass_retained", "output_rel_error", "cache_bytes", "attention_flops", "needle_recall_group1", "needle_recall_group2"):
        vals = np.array([r[key] for r in rows if key in r], dtype=float)
        if len(vals) == 0:
            continue
        agg[f"{key}_mean"] = float(vals.mean())
        agg[f"{key}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        agg[f"{key}_min"] = float(vals.min())
    return agg


def run_grid(
    n_tokens_list: list[int],
    keep_fractions: list[float],
    seeds: list[int],
    policies: list[str] | None = None,
    stride: int = 4,
) -> list[dict]:
    """Full benchmark grid. Returns one aggregated row per cell."""
    names = policies or (BASELINE_NAMES + ["phase_adaptive", "phase_adaptive_norescue"])
    rows: list[dict] = []
    for n_tokens in n_tokens_list:
        for keep_fraction in keep_fractions:
            for name in names:
                rows.append(score_policy(name, n_tokens, keep_fraction, seeds, stride=stride))
    return rows
