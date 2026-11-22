"""MODIFICATION: normalized LMS (NLMS) adaptive noise canceller.

Only ONE line differs from the baseline (src/baseline.py):

    LMS :  w <- w + mu * e_n * x_n
    NLMS:  w <- w + mu * e_n * x_n / (eps + x_n^T x_n)

so the effective step is divided by the energy of the current reference
vector x_n. Everything else (w_0 = 0, residual computed BEFORE the update,
zero reference before the recording, same delay matrix) is identical.

Attribution: standard NLMS; see S. Haykin, Adaptive Filter Theory
(Prentice Hall), chapter on normalized LMS. Implemented independently here.

Setting eps very large makes the NLMS step approximately mu/eps * e_n * x_n,
i.e. plain LMS with a smaller step. This is used for the ablation.
"""
import numpy as np
from src.common import delay_matrix


def nlms(d, r, M, mu, eps=1e-6):
    """Run NLMS on one recording.

    Parameters
    ----------
    d : array (N,)  measured signal (ECG + interference)
    r : array (N,)  noise reference
    M : int >= 1    filter length
    mu : float >= 0 normalized step size (stable roughly for 0 < mu < 2
                    in this half-squared-error convention)
    eps : float >= 0 regularizer that avoids division by zero when the
                    reference energy is tiny

    Returns
    -------
    e : array (N,)  cleaned output (residual)
    w : array (M,)  final coefficients
    Inputs are not modified.
    """
    d = np.asarray(d, dtype=float)
    r = np.asarray(r, dtype=float)
    if d.shape != r.shape or d.ndim != 1:
        raise ValueError("d and r must be 1-D arrays of equal length")
    if M < 1 or int(M) != M:
        raise ValueError("M must be a positive integer")
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    X = delay_matrix(r, int(M))
    energy = np.einsum("ij,ij->i", X, X)     # x_n^T x_n for every n
    w = np.zeros(int(M))
    e = np.empty(len(d))
    with np.errstate(all="ignore"):
        for n in range(len(d)):
            x = X[n]
            e[n] = d[n] - w @ x
            denom = eps + energy[n]
            if denom > 0:                     # eps = 0 and x = 0: skip update
                w = w + mu * e[n] * x / denom
    return e, w