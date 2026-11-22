import numpy as np
import pytest
from src.baseline import lms
from src.common import delay_matrix
from src.data import make_ecg, FS
from src.metrics import snr_improvement_db


def test_delay_matrix_hand_example():
    X = delay_matrix([1.0, 2.0, 3.0], 2)
    assert np.array_equal(X, [[1, 0], [2, 1], [3, 2]])


def test_guide_example():
    # d=[1,1], r=[1,1], M=1, mu=0.5 -> e=[1, 0.5], final w = 0.75
    e, w = lms([1, 1], [1, 1], M=1, mu=0.5)
    assert np.allclose(e, [1.0, 0.5])
    assert np.allclose(w, [0.75])


def test_zero_reference_leaves_input_unchanged():
    d = np.array([0.3, -1.0, 2.0, 0.5])
    e, w = lms(d, np.zeros(4), M=3, mu=0.1)
    assert np.array_equal(e, d)
    assert np.array_equal(w, np.zeros(3))


def test_mu_zero_is_identity():
    d = np.array([1.0, 2.0, 3.0])
    e, w = lms(d, [1.0, 1.0, 1.0], M=2, mu=0.0)
    assert np.array_equal(e, d) and not w.any()


def test_filter_longer_than_signal():
    e, w = lms([1.0, 2.0], [1.0, 1.0], M=10, mu=0.1)
    assert e.shape == (2,) and w.shape == (10,)
    assert np.all(np.isfinite(e))


def test_inputs_not_mutated():
    d = np.arange(10, dtype=float)
    r = np.arange(10, dtype=float)[::-1].copy()
    d0, r0 = d.copy(), r.copy()
    lms(d, r, M=4, mu=0.01)
    assert np.array_equal(d, d0) and np.array_equal(r, r0)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        lms([1, 2], [1], M=1, mu=0.1)
    with pytest.raises(ValueError):
        lms([1, 2], [1, 2], M=0, mu=0.1)
    with pytest.raises(ValueError):
        lms([1, 2], [1, 2], M=1, mu=-0.1)


def test_reduces_noise_on_synthetic_ecg():
    s, d, r = make_ecg(0)
    e, _ = lms(d, r, M=8, mu=0.02)
    assert e.shape == d.shape and np.all(np.isfinite(e))
    assert snr_improvement_db(s, d, e, FS) > 3.0     # clearly better than no filter


def test_diverges_when_mu_far_too_large():
    # Stability needs roughly mu < 2/(M * reference power); mu=5 is far outside.
    _, d, r = make_ecg(0)
    e, _ = lms(d, r, M=8, mu=5.0)
    assert (not np.all(np.isfinite(e))) or np.max(np.abs(e)) > 1e6