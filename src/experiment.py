"""Experiment machinery: validation tuning and the paired test grid.

Both methods receive identical recordings and identical tuning budgets
(same M grid, same number of mu values). Tuning uses validation seeds only.
"""
import time
import numpy as np
from scipy.signal import find_peaks

from src.baseline import lms
from src.modified import nlms
from src.data import make_ecg, FS
from src.metrics import snr_improvement_db, mse, diverged

GEN_KEYS = ("scale", "change_at", "leak")


def mu_grid(spec):
    """Log-spaced step sizes from {"lo":..., "hi":..., "n":...}."""
    return np.logspace(np.log10(spec["lo"]), np.log10(spec["hi"]), spec["n"])


def run_method(method, d, r, M, mu, eps):
    """Return the cleaned output e for 'lms' or 'nlms'."""
    if method == "lms":
        return lms(d, r, M, mu)[0]
    if method == "nlms":
        return nlms(d, r, M, mu, eps)[0]
    raise ValueError(method)


def timed_run(method, d, r, M, mu, eps, repeats):
    """Run once untimed (warm-up), then `repeats` timed runs. Returns (e, median seconds)."""
    e = run_method(method, d, r, M, mu, eps)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run_method(method, d, r, M, mu, eps)
        times.append(time.perf_counter() - t0)
    return e, float(np.median(times))


def gen_kwargs(condition):
    return {k: condition[k] for k in GEN_KEYS if k in condition}


def make_data(seeds, seconds, gen):
    return [make_ecg(s, seconds=seconds, **gen) for s in seeds]


def tune(method, cfg, seeds, gen):
    """Grid search M and mu on validation recordings. Best = highest mean SNR improvement.

    A diverged run scores -1e9, so settings that diverge are never chosen.
    Returns dict(M, mu, val_score).
    """
    warm = int(cfg["warmup_s"] * FS)
    eps = cfg["nlms_eps"]
    grid = mu_grid(cfg["lms_mu_grid" if method == "lms" else "nlms_mu_grid"])
    data = make_data(seeds, cfg["seconds"], gen)
    best = {"M": None, "mu": None, "val_score": -np.inf}
    for M in cfg["M_grid"]:
        for mu in grid:
            scores = []
            for s, d, r in data:
                e = run_method(method, d, r, M, float(mu), eps)
                v = snr_improvement_db(s, d, e, warm)
                scores.append(-1e9 if not np.isfinite(v) else v)
            m = float(np.mean(scores))
            if m > best["val_score"]:
                best = {"M": int(M), "mu": float(mu), "val_score": m}
    return best


def qrs_error(s, e, warm):
    """Mean |e - s| at clean R-peak samples, relative to mean R-peak height (distortion of peaks)."""
    pk, _ = find_peaks(s, height=0.5, distance=int(0.3 * FS))
    pk = pk[pk >= warm]
    if len(pk) == 0 or not np.all(np.isfinite(e[pk])):
        return np.nan
    return float(np.mean(np.abs(e[pk] - s[pk])) / np.mean(s[pk]))


def run_grid(cfg, log=print):
    """Full protocol. Returns (rows, tuned) where rows is a list of dicts (long format)."""
    warm = int(cfg["warmup_s"] * FS)
    eps = cfg["nlms_eps"]
    val = range(*cfg["val_seeds"])
    test = range(*cfg["test_seeds"])

    base_gen = {"scale": cfg["tune_scale"]}
    tuned = {"fixed": {m: tune(m, cfg, val, base_gen) for m in ("lms", "nlms")},
             "retuned": {}}
    log(f"tuned at scale {cfg['tune_scale']}: {tuned['fixed']}")

    rows = []
    for cond in cfg["conditions"]:
        gen = gen_kwargs(cond)
        retuned = {m: tune(m, cfg, val, gen) for m in ("lms", "nlms")}
        tuned["retuned"][cond["name"]] = retuned
        log(f"[{cond['name']}] retuned: {retuned}")

        variants = [("lms", "lms", tuned["fixed"]["lms"]),
                    ("nlms", "nlms", tuned["fixed"]["nlms"]),
                    ("lms_retuned", "lms", retuned["lms"]),
                    ("nlms_retuned", "nlms", retuned["nlms"])]
        for seed in test:
            s, d, r = make_ecg(seed, seconds=cfg["seconds"], **gen)
            for label, method, p in variants:
                e, t = timed_run(method, d, r, p["M"], p["mu"], eps, cfg["timing_repeats"])
                div = diverged(e)
                rows.append({
                    "seed": seed, "condition": cond["name"], "method": label,
                    "M": p["M"], "mu": p["mu"],
                    "snr_improvement_db": snr_improvement_db(s, d, e, warm),
                    "mse": mse(s[warm:], e[warm:]),
                    "qrs_error": np.nan if div else qrs_error(s, e, warm),
                    "diverged": div, "runtime_s": t,
                })
    return rows, tuned