"""Helpers shared by the baseline (LMS) and the modification (NLMS)."""
import numpy as np


def delay_matrix(r, M):
    """Delayed-reference matrix X, shape (N, M), with X[n, j] = r[n-j] (0 if n-j < 0).

    Row n is the vector x_n = [r_n, r_{n-1}, ..., r_{n-M+1}] from the guide,
    with reference samples before the recording set to zero.
    """
    r = np.asarray(r, dtype=float)
    N = len(r)
    X = np.zeros((N, M))
    for j in range(min(M, N)):
        X[j:, j] = r[:N - j]
    return X