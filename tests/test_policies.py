import numpy as np
import pytest

from phasekv.attention import make_context
from phasekv.policies import POLICIES, apply_policy


@pytest.mark.parametrize("name", list(POLICIES))
def test_policies_keep_sinks_and_have_expected_shape(name):
    ctx = make_context(seed=0, n_tokens=128, n_heads=4)
    mask = apply_policy(name, ctx, 0.25, seed=0)
    assert mask.shape == (4, 128)
    assert mask.dtype == bool
    # Sinks are always kept by every policy.
    assert mask[:, ctx.sink_mask].all()


@pytest.mark.parametrize("name", ["random", "streaming", "snapkv", "knorm", "expected_attention", "tova"])
def test_scorer_policies_keep_exact_budget(name):
    ctx = make_context(seed=0, n_tokens=128, n_heads=4)
    mask = apply_policy(name, ctx, 0.25, seed=0)
    assert (mask.sum(axis=1) == 32).all()


def test_chunked_keeps_one_per_block_plus_sinks():
    ctx = make_context(seed=0, n_tokens=128, n_heads=2)
    mask = apply_policy("chunked", ctx, 0.25)
    # 32 blocks of 4 -> at least 32 kept per head (sinks may add).
    assert (mask.sum(axis=1) >= 32).all()
    assert (mask.sum(axis=1) <= 36).all()


def test_random_policy_deterministic_for_seed():
    ctx = make_context(seed=0, n_tokens=64)
    a = apply_policy("random", ctx, 0.5, seed=42)
    b = apply_policy("random", ctx, 0.5, seed=42)
    assert np.array_equal(a, b)


def test_snapkv_beats_random_on_needle_recall():
    ctx = make_context(seed=5, n_tokens=512)
    snap = apply_policy("snapkv", ctx, 0.25, seed=5)
    rand = apply_policy("random", ctx, 0.25, seed=5)
    assert snap[:, ctx.needle_mask].mean() > rand[:, ctx.needle_mask].mean()


def test_unknown_policy_raises():
    ctx = make_context(seed=0, n_tokens=32)
    with pytest.raises(ValueError):
        apply_policy("nope", ctx, 0.5)
