"""Quality metrics for noise cancellation. Larger SNR is better; smaller MSE is better."""
import numpy as np


def mse(s, s_hat):
    with np.errstate(over="ignore", invalid="ignore"):
        err = np.sum((np.asarray(s_hat) - np.asarray(s)) ** 2)
    if not np.isfinite(err):
        return np.nan
    return err / len(s)


def snr_db(s, s_hat):
    """SNR in dB = 10 log10( sum s^2 / sum (s_hat - s)^2 ).

    NaN if s_hat is non-finite (diverged filter); +inf if the error is exactly 0.
    """
    s = np.asarray(s)
    with np.errstate(over="ignore", invalid="ignore"):
        err = np.sum((np.asarray(s_hat) - s) ** 2)
        power = np.sum(s ** 2)
    if not np.isfinite(err):
        return np.nan
    if err == 0:
        return np.inf
    return 10.0 * np.log10(power / err)


def snr_improvement_db(clean, measured, e, warmup):
    """PRIMARY METRIC: SNR(e) - SNR(measured) in dB, evaluated after `warmup` samples.

    warmup : int, number of initial samples excluded (1 s = fs samples).
    Returns NaN if the filter output is non-finite.
    """
    s, d, y = clean[warmup:], measured[warmup:], e[warmup:]
    return snr_db(s, y) - snr_db(s, d)


def diverged(e):
    """True if the output contains NaN/inf."""
    return not bool(np.all(np.isfinite(e)))