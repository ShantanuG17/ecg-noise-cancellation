"""Paired statistics over recordings (the unit of replication is one recording)."""
import numpy as np
import pandas as pd
from scipy import stats


def paired_ci(diff, conf=0.95, n_boot=10000, seed=0):
    """Mean of paired differences with t-interval and bootstrap percentile interval.

    Non-finite differences (diverged runs) are dropped; n reports how many remain.
    """
    d = np.asarray(diff, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    out = {"n": n, "mean": np.nan, "t_lo": np.nan, "t_hi": np.nan,
           "boot_lo": np.nan, "boot_hi": np.nan, "frac_positive": np.nan}
    if n < 2:
        return out
    mean = d.mean()
    h = stats.t.ppf(0.5 + conf / 2, n - 1) * d.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(n_boot, n)).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * (1 - conf) / 2, 100 * (1 + conf) / 2])
    out.update(mean=mean, t_lo=mean - h, t_hi=mean + h, boot_lo=lo, boot_hi=hi,
               frac_positive=float((d > 0).mean()))
    return out


def paired_table(df, a, b, metric):
    """Per condition: mean of (metric[a] - metric[b]) over test recordings, with CIs."""
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        p = g.pivot(index="seed", columns="method", values=metric)
        diff = (p[a] - p[b])
        r = paired_ci(diff.values)
        r.update(condition=cond, a=a, b=b, metric=metric, n_total=len(p),
                 n_diverged_a=int(g[g.method == a].diverged.sum()),
                 n_diverged_b=int(g[g.method == b].diverged.sum()))
        rows.append(r)
    cols = ["condition", "a", "b", "metric", "n", "n_total", "n_diverged_a",
            "n_diverged_b", "mean", "t_lo", "t_hi", "boot_lo", "boot_hi", "frac_positive"]
    return pd.DataFrame(rows)[cols]