import numpy as np

from phasekv.attention import make_context
from phasekv.cache import attention_flops, cache_bytes, keep_fraction
from phasekv.policies import apply_policy
from phasekv.probes import evaluate


def test_evaluate_full_cache_is_lossless():
    ctx = make_context(seed=0, n_tokens=128)
    keep = np.ones_like(ctx.keys[:, :, 0], dtype=bool)
    metrics = evaluate(ctx, keep)
    assert metrics["output_rel_error"] <= 1e-12
    assert metrics["needle_recall"] == 1.0
    assert metrics["mass_retained"] == 1.0


def test_cache_accounting_scales_with_kept():
    ctx = make_context(seed=0, n_tokens=128, n_heads=2)
    half = np.zeros((2, 128), dtype=bool)
    half[:, :64] = True
    assert cache_bytes(half, 32, 32) == 64 * 2 * 64 * 2
    assert keep_fraction(half) == 0.5
    assert attention_flops(4, half.sum(axis=1), 32) == 4 * 128 * 32


def test_error_decreases_with_budget():
    ctx = make_context(seed=4, n_tokens=256)
    errors = []
    for keep in (0.125, 0.25, 0.5):
        mask = apply_policy("snapkv", ctx, keep, seed=4)
        errors.append(evaluate(ctx, mask)["output_rel_error"])
    assert errors[0] >= errors[1] >= errors[2]


def test_eval_queries_not_used_for_scoring():
    # Changing eval queries must not change any policy's keep mask:
    # policies score from observation queries only.
    ctx = make_context(seed=6, n_tokens=128)
    mask = apply_policy("snapkv", ctx, 0.25, seed=6)
    ctx2 = make_context(seed=6, n_tokens=128)
    ctx2.eval_queries = ctx2.eval_queries + 10.0
    mask2 = apply_policy("snapkv", ctx2, 0.25, seed=6)
    assert np.array_equal(mask, mask2)
