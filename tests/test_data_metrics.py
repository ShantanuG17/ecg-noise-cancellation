import numpy as np
from src.data import make_ecg, SPLITS, FS
from src.metrics import mse, snr_db, snr_improvement_db, diverged


def test_shapes_and_finite():
    s, d, r = make_ecg(0)
    assert s.shape == d.shape == r.shape == (12 * FS,)
    assert np.all(np.isfinite(s)) and np.all(np.isfinite(d)) and np.all(np.isfinite(r))


def test_deterministic_and_seed_dependent():
    a = make_ecg(3)
    b = make_ecg(3)
    c = make_ecg(4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[1], c[1])


def test_splits_disjoint():
    seen = set()
    for seeds in SPLITS.values():
        assert not (seen & set(seeds))
        seen |= set(seeds)


def test_scale_only_changes_reference_path():
    s1, _, r1 = make_ecg(5, scale=1.0)
    s2, _, r2 = make_ecg(5, scale=2.0)
    assert np.array_equal(s1, s2)           # clean unaffected
    assert np.allclose(r2, 2.0 * r1)


def test_pathway_change_only_after_switch():
    _, d0, _ = make_ecg(7)
    _, d1, _ = make_ecg(7, change_at=6.0)
    n = 6 * FS
    assert np.array_equal(d0[:n], d1[:n])
    assert not np.array_equal(d0[n:], d1[n:])


def test_leakage_changes_reference_not_measured():
    s, d0, r0 = make_ecg(9)
    _, d1, r1 = make_ecg(9, leak=0.2)
    assert np.array_equal(d0, d1)
    assert np.allclose(r1, r0 + 0.2 * s)


def test_metrics_hand_values():
    s = np.array([1.0, 1.0])
    assert mse(s, np.array([0.0, 0.0])) == 1.0
    assert np.isinf(snr_db(s, s))
    assert np.isclose(snr_db(s, np.array([0.0, 0.0])), 0.0)   # error power = signal power
    assert np.isnan(snr_db(s, np.array([np.nan, 0.0])))
    assert diverged(np.array([1.0, np.inf]))


def test_improvement_of_input_with_itself_is_zero():
    s, d, _ = make_ecg(0)
    assert np.isclose(snr_improvement_db(s, d, d, FS), 0.0)