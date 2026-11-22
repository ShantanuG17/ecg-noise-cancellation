"""BASELINE: least-mean-squares (LMS) adaptive noise canceller.

Model (course guide / Thakor & Zhu 1991, reference-noise cancellation):
    y_n = w_n^T x_n          prediction of the interference
    e_n = d_n - y_n          cleaned output (the residual, NOT y_n)
    w_{n+1} = w_n + mu * e_n * x_n
with w_0 = 0. The residual is computed BEFORE the update. This is the
half-squared-error convention (no factor of 2 in the update).

This file is frozen once the baseline commit is tagged; do not edit it
when developing the modification.
"""
import numpy as np
from src.common import delay_matrix


def lms(d, r, M, mu):
    """Run LMS on one recording.

    Parameters
    ----------
    d : array (N,)  measured signal (ECG + interference)
    r : array (N,)  noise reference
    M : int >= 1    filter length
    mu : float >= 0 step size

    Returns
    -------
    e : array (N,)  cleaned output (residual)
    w : array (M,)  final coefficients
    Inputs are not modified. e may contain inf/nan if the filter diverges.
    """
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    if d.shape != r.shape or d.ndim != 1:
        raise ValueError("d and r must be 1-D arrays of equal length")
    if M < 1 or int(M) != M:
        raise ValueError("M must be a positive integer")
    if mu < 0:
        raise ValueError("mu must be non-negative")

    X = delay_matrix(r, int(M))
    w = np.zeros(int(M))
    e = np.empty(len(d))
    with np.errstate(all="ignore"):          # let divergence show up as inf/nan
        for n in range(len(d)):
            x = X[n]
            e[n] = d[n] - w @ x
            w = w + mu * e[n] * x
    return e, w