import numpy as np

from phasekv.attention import make_context


def test_context_shapes_and_determinism():
    a = make_context(seed=7, n_tokens=128, n_heads=4)
    b = make_context(seed=7, n_tokens=128, n_heads=4)
    assert a.keys.shape == (4, 128, 32)
    assert a.values.shape == (4, 128, 32)
    assert np.array_equal(a.keys, b.keys)
    assert a.needle_mask.sum() == 16
    assert a.sink_mask.sum() == 4
    assert not (a.needle_mask & a.sink_mask).any()


def test_attention_rows_sum_to_one():
    ctx = make_context(seed=1, n_tokens=64, n_heads=2)
    w = ctx.attention(ctx.obs_queries)
    assert w.shape == (2, 8, 64)
    np.testing.assert_allclose(w.sum(axis=-1), 1.0, atol=1e-9)


def test_full_output_matches_unmasked_output():
    ctx = make_context(seed=2, n_tokens=64, n_heads=2)
    keep = np.ones((2, 64), dtype=bool)
    np.testing.assert_allclose(
        ctx.output(ctx.eval_queries, keep), ctx.full_output(ctx.eval_queries), atol=1e-9
    )


def test_needles_spread_across_phases():
    ctx = make_context(seed=3, n_tokens=256)
    phases = np.arange(256) % 4
    for phase in range(4):
        assert (ctx.needle_mask & (phases == phase)).sum() >= 1
