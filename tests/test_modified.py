import numpy as np
import pytest
from src.baseline import lms
from src.modified import nlms
from src.data import make_ecg, FS
from src.metrics import snr_improvement_db


def test_hand_example_nlms_vs_lms():
    # d=[1,1], r=[2,2], M=1, mu=0.5, eps=0
    # NLMS: e=[1, 0.5], w=0.375.   LMS: e=[1, -1], w=0 (step too big for scale 2)
    e, w = nlms([1, 1], [2, 2], M=1, mu=0.5, eps=0.0)
    assert np.allclose(e, [1.0, 0.5])
    assert np.allclose(w, [0.375])
    e2, w2 = lms([1, 1], [2, 2], M=1, mu=0.5)
    assert np.allclose(e2, [1.0, -1.0])
    assert np.allclose(w2, [0.0])


def test_zero_reference_leaves_input_unchanged():
    d = np.array([0.3, -1.0, 2.0, 0.5])
    for eps in (0.0, 1e-6):
        e, w = nlms(d, np.zeros(4), M=3, mu=0.5, eps=eps)
        assert np.array_equal(e, d)
        assert not w.any()


def test_mu_zero_is_identity():
    d = np.array([1.0, 2.0, 3.0])
    e, _ = nlms(d, [1.0, 1.0, 1.0], M=2, mu=0.0)
    assert np.array_equal(e, d)


def test_scale_invariance_with_eps_zero():
    _, d, r = make_ecg(1)
    e1, _ = nlms(d, r, M=8, mu=0.5, eps=0.0)
    e2, _ = nlms(d, 7.0 * r, M=8, mu=0.5, eps=0.0)
    assert np.allclose(e1, e2, atol=1e-9)


def test_lms_is_not_scale_invariant():
    _, d, r = make_ecg(1)
    e1, _ = lms(d, r, M=8, mu=0.01)
    e2, _ = lms(d, 3.0 * r, M=8, mu=0.01)
    assert not np.allclose(e1, e2, atol=1e-6)


def test_huge_eps_matches_lms_with_scaled_step():
    # eps >> x^T x  =>  NLMS(mu, eps) ~ LMS(mu/eps)
    _, d, r = make_ecg(2)
    eps = 1e8
    e_n, _ = nlms(d, r, M=8, mu=0.02 * eps, eps=eps)
    e_l, _ = lms(d, r, M=8, mu=0.02)
    assert np.allclose(e_n, e_l, rtol=1e-4, atol=1e-6)


def test_filter_longer_than_signal_and_shapes():
    e, w = nlms([1.0, 2.0], [1.0, 1.0], M=10, mu=0.5)
    assert e.shape == (2,) and w.shape == (10,)
    assert np.all(np.isfinite(e))


def test_inputs_not_mutated():
    d = np.arange(10, dtype=float)
    r = np.arange(10, dtype=float)[::-1].copy()
    d0, r0 = d.copy(), r.copy()
    nlms(d, r, M=4, mu=0.5)
    assert np.array_equal(d, d0) and np.array_equal(r, r0)


def test_bad_arguments_raise():
    with pytest.raises(ValueError):
        nlms([1, 2], [1], M=1, mu=0.5)
    with pytest.raises(ValueError):
        nlms([1, 2], [1, 2], M=0, mu=0.5)
    with pytest.raises(ValueError):
        nlms([1, 2], [1, 2], M=1, mu=-0.5)
    with pytest.raises(ValueError):
        nlms([1, 2], [1, 2], M=1, mu=0.5, eps=-1.0)


def test_reduces_noise_on_synthetic_ecg_at_several_scales():
    # NLMS with one fixed mu: clear gain where interference is substantial.
    for scale in (1.0, 10.0):
        s, d, r = make_ecg(0, scale=scale)
        e, _ = nlms(d, r, M=8, mu=0.5)
        assert np.all(np.isfinite(e))
        assert snr_improvement_db(s, d, e, FS) > 3.0


def test_small_scale_stays_finite_and_bounded():
    # Scale 0.1: interference is weaker than the ECG-driven adaptation noise,
    # so gain can be negative. This is a documented limitation, not a bug.
    s, d, r = make_ecg(0, scale=0.1)
    e, _ = nlms(d, r, M=8, mu=0.5)
    assert np.all(np.isfinite(e))
    assert np.max(np.abs(e)) < 10 * np.max(np.abs(d))