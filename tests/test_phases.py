import numpy as np

from phasekv.phases import phase_of, phase_recall, stratified_topk, worst_phase_recall


def test_phase_of():
    assert list(phase_of(np.arange(8), 4)) == [0, 1, 2, 3, 0, 1, 2, 3]
    try:
        phase_of(np.arange(3), 0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("stride 0 must raise")


def test_stratified_topk_guarantees_per_phase_quota():
    rng = np.random.default_rng(0)
    scores = rng.random(64)
    # All score mass in phase 0: a plain top-k would starve other phases.
    scores[1::4] = 0.0
    scores[2::4] = 0.0
    scores[3::4] = 0.0
    protected = np.zeros(64, dtype=bool)
    keep = stratified_topk(scores, 16, 4, protected)
    assert keep.sum() == 16
    phases = phase_of(np.arange(64), 4)
    for phase in range(4):
        assert keep[phases == phase].sum() >= 1


def test_stratified_topk_respects_protected_and_budget():
    scores = np.arange(32, dtype=float)
    protected = np.zeros(32, dtype=bool)
    protected[:4] = True
    keep = stratified_topk(scores, 10, 4, protected)
    assert keep.sum() == 10
    assert keep[:4].all()


def test_phase_recall_and_worst_phase():
    keep = np.zeros((2, 8), dtype=bool)
    keep[:, ::2] = True  # keep only even positions
    needles = np.ones(8, dtype=bool)
    recalls = phase_recall(keep, needles, 4)
    assert recalls == {0: 1.0, 1: 0.0, 2: 1.0, 3: 0.0}
    assert worst_phase_recall(keep, needles, 4) == 0.0
    assert worst_phase_recall(keep, np.zeros(8, dtype=bool), 4) == 1.0
