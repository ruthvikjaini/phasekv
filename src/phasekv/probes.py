"""Evaluation: recall, attention-mass retention, and output fidelity.

All metrics use eval queries only, which no policy sees during scoring
or rescue. Metrics are per context; aggregation lives in scoring.py.
"""

from __future__ import annotations

import numpy as np

from phasekv.attention import AttentionContext
from phasekv.cache import attention_flops, cache_bytes
from phasekv.phases import phase_recall, worst_phase_recall


def evaluate(
    context: AttentionContext,
    keep_mask: np.ndarray,
    stride: int = 4,
    merged_values: np.ndarray | None = None,
) -> dict:
    """Score one compressed cache against the full cache."""
    eval_attn = context.attention(context.eval_queries)
    needle = context.needle_mask
    # Needle recall: fraction of needle (head, token) pairs retained.
    needle_recall = float(keep_mask[:, needle].mean()) if needle.any() else 1.0
    # Attention-mass retention: eval attention mass landing on kept pairs.
    kept_mass = np.where(keep_mask[:, None, :], eval_attn, 0.0).sum(axis=-1)
    mass_retained = float(kept_mass.mean())
    # Output fidelity: relative error of compressed vs full output.
    full = context.full_output(context.eval_queries)
    if merged_values is not None:
        # Recompute output with merged values but the same kept keys.
        weights = eval_attn
        masked = np.where(keep_mask[:, None, :], weights, 0.0)
        denom = masked.sum(axis=-1, keepdims=True)
        safe = np.where(denom > 0, masked / np.where(denom == 0, 1.0, denom), 0.0)
        compressed = np.einsum("hqk,hkd->hqd", safe, merged_values)
    else:
        compressed = context.output(context.eval_queries, keep_mask)
    denom = float(np.linalg.norm(full))
    rel_error = float(np.linalg.norm(full - compressed) / denom) if denom > 0 else 0.0
    per_phase = phase_recall(keep_mask, needle, stride)
    kept_per_head = keep_mask.sum(axis=1)
    result = {
        "needle_recall": needle_recall,
        "worst_phase_recall": worst_phase_recall(keep_mask, needle, stride),
        "per_phase_recall": {str(k): float(v) for k, v in per_phase.items()},
        "mass_retained": mass_retained,
        "output_rel_error": rel_error,
        "keep_fraction": float(keep_mask.mean()),
        "kept_per_head_mean": float(kept_per_head.mean()),
        "cache_bytes": cache_bytes(keep_mask, context.d_k, context.d_v),
        "attention_flops": attention_flops(
            context.eval_queries.shape[1], kept_per_head, context.d_k
        ),
    }
    if context.needle_group is not None and len(context.needle_group) == len(needle):
        for grp in (1, 2):
            sel = context.needle_group == grp
            if sel.any():
                result[f"needle_recall_group{grp}"] = float(keep_mask[:, sel].mean())
    return result
