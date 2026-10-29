import numpy as np

from phasekv.adaptive import PhaseAdaptivePolicy, _head_budgets, merge_downgrade
from phasekv.attention import make_context
from phasekv.policies import apply_policy
from phasekv.probes import evaluate


def test_head_budgets_sum_and_bounds():
    rng = np.random.default_rng(0)
    obs = rng.random((8, 4, 128))
    obs = obs / obs.sum(axis=-1, keepdims=True)
    budgets = _head_budgets(obs, 256, 8)
    assert budgets.sum() == 256
    assert (budgets >= 8).all()  # floor = 25% of uniform 32
    assert (budgets <= 64).all()


def test_adaptive_preserves_total_budget_after_rescue():
    ctx = make_context(seed=0, n_tokens=512)
    result = PhaseAdaptivePolicy(stride=4).apply(ctx, 0.25)
    assert result.keep_mask.sum() == 8 * 128
    assert result.head_budgets.sum() == 8 * 128
    assert result.keep_mask[:, ctx.sink_mask].all()
    assert 0.0 <= result.rescue_recall_before <= 1.0
    assert result.rescue_recall_after >= result.rescue_recall_before - 1e-12
    assert result.n_restores >= 0


def test_adaptive_worst_phase_at_least_snapkv_on_average():
    # Over 5 seeds the phase-stratified policy must not lose to SnapKV
    # on the metric it is designed for: worst-phase needle recall.
    worst_adaptive, worst_snap = [], []
    for seed in range(5):
        ctx = make_context(seed=seed, n_tokens=512)
        adaptive = PhaseAdaptivePolicy(stride=4).apply(ctx, 0.25)
        snap = apply_policy("snapkv", ctx, 0.25, seed=seed)
        worst_adaptive.append(evaluate(ctx, adaptive.keep_mask)["worst_phase_recall"])
        worst_snap.append(evaluate(ctx, snap)["worst_phase_recall"])
    assert float(np.mean(worst_adaptive)) >= float(np.mean(worst_snap))


def test_adaptive_recall_at_least_chunked_on_average():
    recalls_a, recalls_c = [], []
    for seed in range(5):
        ctx = make_context(seed=seed, n_tokens=512)
        adaptive = PhaseAdaptivePolicy(stride=4).apply(ctx, 0.25)
        chunked = apply_policy("chunked", ctx, 0.25)
        recalls_a.append(evaluate(ctx, adaptive.keep_mask)["needle_recall"])
        recalls_c.append(evaluate(ctx, chunked)["needle_recall"])
    assert float(np.mean(recalls_a)) >= float(np.mean(recalls_c))


def test_adaptive_recovers_group_b_needles_that_snapkv_misses():
    # Group B is what later queries ask about; SnapKV scores from the
    # observation window (group A), so it should lose group B, while
    # rescue probes recover a large share of it.
    b_adaptive, b_snap = [], []
    for seed in range(5):
        ctx = make_context(seed=seed, n_tokens=512)
        adaptive = PhaseAdaptivePolicy(stride=4).apply(ctx, 0.25)
        snap = apply_policy("snapkv", ctx, 0.25, seed=seed)
        b_adaptive.append(evaluate(ctx, adaptive.keep_mask)["needle_recall_group2"])
        b_snap.append(evaluate(ctx, snap)["needle_recall_group2"])
    assert float(np.mean(b_adaptive)) > float(np.mean(b_snap)) + 0.15
    assert float(np.mean(b_adaptive)) >= 0.5


def test_rescue_is_what_recovers_group_b():
    from phasekv.adaptive import PhaseAdaptivePolicy as PAP

    with_rescue, without = [], []
    for seed in range(3):
        ctx = make_context(seed=seed, n_tokens=512)
        on = PAP(stride=4, recall_threshold=0.95).apply(ctx, 0.25)
        off = PAP(stride=4, recall_threshold=0.0).apply(ctx, 0.25)
        with_rescue.append(evaluate(ctx, on.keep_mask)["needle_recall_group2"])
        without.append(evaluate(ctx, off.keep_mask)["needle_recall_group2"])
    assert float(np.mean(with_rescue)) > float(np.mean(without)) + 0.15


def test_merge_downgrade_reduces_output_error():
    ctx = make_context(seed=3, n_tokens=256)
    mask = apply_policy("snapkv", ctx, 0.25, seed=3)
    plain = evaluate(ctx, mask)["output_rel_error"]
    merged, _ = merge_downgrade(ctx, mask)
    with_merge = evaluate(ctx, mask, merged_values=merged)["output_rel_error"]
    assert with_merge <= plain + 1e-12
    # Merged values keep the cache at the kept size.
    assert merged.shape == ctx.values.shape


def test_rescue_never_exceeds_budget_with_tight_threshold():
    ctx = make_context(seed=9, n_tokens=256)
    result = PhaseAdaptivePolicy(stride=4, recall_threshold=1.0, max_rescue_swaps=10).apply(ctx, 0.125)
    assert result.keep_mask.sum() == 8 * 32
    assert result.n_restores <= 10 * 8
