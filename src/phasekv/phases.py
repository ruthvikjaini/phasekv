"""Compression phases: the positional coordinate chunked compression adds.

Following Zhu et al. (arXiv:2609.36322), a token's *phase* is its
position relative to compression-window boundaries. Chunked pooling
blends tokens inside each window, so a fact's retrievability can depend
systematically on its phase even when average recall looks fine. This
module computes phases and per-phase recall, and allocates a keep
budget fairly across phase classes.
"""

from __future__ import annotations

import numpy as np


def phase_of(positions: np.ndarray, stride: int) -> np.ndarray:
    """Phase of each position: position modulo the compression stride."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    return np.asarray(positions) % stride


def phase_recall(
    keep_mask: np.ndarray,
    needle_mask: np.ndarray,
    stride: int,
) -> dict[int, float]:
    """Per-phase needle recall, averaged over heads.

    Returns {phase: recall}. Phases with no needles are omitted.
    """
    phases = phase_of(np.arange(len(needle_mask)), stride)
    out: dict[int, float] = {}
    for phase in sorted(set(int(p) for p in phases)):
        sel = needle_mask & (phases == phase)
        if not sel.any():
            continue
        out[phase] = float(keep_mask[:, sel].mean())
    return out


def worst_phase_recall(
    keep_mask: np.ndarray,
    needle_mask: np.ndarray,
    stride: int,
) -> float:
    """Minimum per-phase needle recall; 1.0 when no needles exist."""
    recalls = phase_recall(keep_mask, needle_mask, stride)
    if not recalls:
        return 1.0
    return min(recalls.values())


def stratified_topk(
    scores: np.ndarray,
    budget: int,
    stride: int,
    protected: np.ndarray,
) -> np.ndarray:
    """Top-k selection with a per-phase quota.

    ``scores`` is (n_tokens,) for one head. ``protected`` positions are
    always kept and do not consume budget. The remaining budget is split
    as evenly as possible across phase classes, then filled by score
    within each class; any leftover budget goes to the highest-scored
    unkept positions overall. Guarantees no phase class is starved when
    the budget allows at least one slot per class.
    """
    n = len(scores)
    keep = np.zeros(n, dtype=bool)
    keep[protected] = True
    remaining = max(0, budget - int(protected.sum()))
    if remaining == 0:
        return keep
    phases = phase_of(np.arange(n), stride)
    candidates = np.where(~keep)[0]
    order = candidates[np.argsort(-scores[candidates], kind="stable")]
    # Phase quotas: round-robin over phases in score order.
    by_phase: dict[int, list[int]] = {}
    for idx in order:
        by_phase.setdefault(int(phases[idx]), []).append(int(idx))
    quota = {phase: len(members) for phase, members in by_phase.items()}
    total = sum(quota.values())
    allocated = {phase: 0 for phase in quota}
    picked: list[int] = []
    # Round-robin: one pick per phase per round, highest score first.
    pointers = {phase: 0 for phase in quota}
    while len(picked) < remaining and total > 0:
        progressed = False
        for phase in sorted(by_phase):
            if len(picked) >= remaining:
                break
            members = by_phase[phase]
            ptr = pointers[phase]
            if ptr < len(members):
                picked.append(members[ptr])
                pointers[phase] += 1
                allocated[phase] += 1
                progressed = True
        if not progressed:
            break
    for idx in picked:
        keep[idx] = True
    return keep
