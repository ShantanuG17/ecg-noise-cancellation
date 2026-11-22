"""Synthetic ECG-like data for reference-noise cancellation (Track B)."""
import numpy as np

FS = 250  # sample rate, samples per second

# Interference path (guide). 'PATHWAY_B' is the changed path used after the switch.
PATHWAY = np.array([0.35, -0.15, 0.08])
PATHWAY_B = np.array([-0.20, 0.30, -0.10])

# Whole-recording seed splits (never split neighbouring samples).
SPLITS = {
    "dev": list(range(0, 20)),
    "val": list(range(100, 110)),
    "test": list(range(1000, 1020)),
}


def make_ecg(seed, seconds=12, fs=FS, scale=1.0, change_at=None, leak=0.0):
    """Generate one synthetic recording.

    Parameters
    ----------
    seed : int           RNG seed (one seed = one whole recording).
    seconds : float      Recording length in seconds.
    fs : int             Sample rate in Hz.
    scale : float        Reference amplitude multiplier (reference scale).
    change_at : float|None
        If given, time in seconds at which the interference pathway switches
        from PATHWAY to PATHWAY_B (non-stationary interference path).
    leak : float
        Leakage: reference <- reference + leak * clean (guide case: 0.2).
        Applied after the interference is formed, so the interference itself
        is unchanged.

    Returns
    -------
    clean, measured, reference : ndarray, each shape (int(seconds*fs),)
        clean     s_n  true ECG-like signal (never given to the filters)
        measured  d_n  = s_n + interference + small sensor noise
        reference r_n  noise reference
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * fs)) / fs
    clean = np.zeros_like(t)
    interval = rng.uniform(0.75, 1.05)
    pulses = [(0.12, -0.18, 0.035),
              (-0.15, -0.035, 0.012),
              (1.0, 0.0, 0.016),
              (-0.25, 0.035, 0.014),
              (0.30, 0.24, 0.065)]
    for beat in np.arange(0.6, seconds - 0.4, interval):
        for amplitude, offset, width in pulses:
            clean += amplitude * np.exp(
                -0.5 * ((t - beat - offset) / width) ** 2)

    phase = rng.uniform(0, 2 * np.pi)
    reference = scale * (np.sin(2 * np.pi * 50 * t + phase)
                         + 0.2 * rng.normal(size=len(t)))

    interference = np.convolve(reference, PATHWAY, mode="full")[:len(t)]
    if change_at is not None:
        n_switch = int(change_at * fs)
        after = np.convolve(reference, PATHWAY_B, mode="full")[:len(t)]
        interference = np.where(np.arange(len(t)) < n_switch,
                                interference, after)

    measured = clean + interference
    measured += 0.02 * rng.normal(size=len(t))  # same draw order as the guide

    if leak:
        reference = reference + leak * clean
    return clean, measured, reference