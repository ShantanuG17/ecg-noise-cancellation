"""Tables and figures from saved results only.

Usage: python analyze_results.py --results results/final
Writes CSV tables to the results folder and PNG figures to <results>/figures/.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis import paired_table
from src.data import make_ecg, FS
from src.experiment import run_method, gen_kwargs


def forest(main, abl, path):
    conds = list(main.condition)
    y = np.arange(len(conds))[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for tab, off, lab, col in ((main, 0.12, "tuned at scale 1 (mismatch)", "tab:blue"),
                               (abl, -0.12, "retuned per condition (ablation)", "tab:orange")):
        ax.errorbar(tab["mean"], y + off,
                    xerr=[tab["mean"] - tab["t_lo"], tab["t_hi"] - tab["mean"]],
                    fmt="o", color=col, capsize=3, label=lab)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(conds)
    ax.set_xlabel("Paired difference in SNR improvement, NLMS - LMS (dB); >0 favours NLMS")
    ax.set_title("Mean paired difference with 95% t-interval (20 test recordings per condition)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def example_figure(cfg, tuned, path, names, seed=1000, t0=2.0, t1=6.0):
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 3.2 * len(names)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cond = next(c for c in cfg["conditions"] if c["name"] == name)
        s, d, r = make_ecg(seed, seconds=cfg["seconds"], **gen_kwargs(cond))
        t = np.arange(len(s)) / FS
        m = (t >= t0) & (t < t1)
        ax.plot(t[m], s[m], "k", lw=1.5, label="clean ECG (truth)")
        for method, col in (("lms", "tab:red"), ("nlms", "tab:blue")):
            p = tuned["fixed"][method]
            e = run_method(method, d, r, p["M"], p["mu"], cfg["nlms_eps"])
            lab = f"{method.upper()} (M={p['M']}, mu={p['mu']:.3g})"
            if not np.all(np.isfinite(e)):
                lab += " DIVERGED"
            ax.plot(t[m], e[m], color=col, lw=0.8, alpha=0.8, label=lab)
        ax.set_ylim(-0.6, 1.4)
        ax.set_ylabel("amplitude (a.u.)")
        ax.set_title(f"{name}, test seed {seed}")
        ax.legend(loc="upper right", fontsize=7)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def boxes(df, conds, methods, metric, ylabel, ax):
    pos, data, labels = [], [], []
    k = 0
    for c in conds:
        for m in methods:
            v = df[(df.condition == c) & (df.method == m)][metric].values
            data.append(v[np.isfinite(v)])
            n_div = int(df[(df.condition == c) & (df.method == m)].diverged.sum())
            labels.append(f"{c}\n{m}" + (f"\n({n_div} diverged)" if n_div else ""))
            pos.append(k)
            k += 1
        k += 0.6
    ax.boxplot(data, positions=pos, widths=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(ylabel)


def stress_figure(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    boxes(df, ["stress_scale10", "stress_scale0.1", "leakage"], ["lms", "nlms"],
          "snr_improvement_db", "SNR improvement (dB), finite runs only", axes[0])
    axes[0].set_title("Stress conditions: SNR improvement")
    boxes(df, ["scale1", "leakage"], ["lms", "nlms"], "qrs_error",
          "R-peak error / mean R-peak height", axes[1])
    axes[1].set_title("Peak distortion (lower is better)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()
    R = a.results
    df = pd.read_csv(os.path.join(R, "results.csv"))
    with open(os.path.join(R, "config_used.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(R, "tuned.json")) as f:
        tuned = json.load(f)
    figdir = os.path.join(R, "figures")
    os.makedirs(figdir, exist_ok=True)

    main_t = paired_table(df, "nlms", "lms", "snr_improvement_db")
    abl_t = paired_table(df, "nlms_retuned", "lms_retuned", "snr_improvement_db")
    qrs_t = paired_table(df, "nlms", "lms", "qrs_error")
    df["log_runtime"] = np.log(df["runtime_s"])
    rt = paired_table(df, "nlms", "lms", "log_runtime")
    for c in ("mean", "t_lo", "t_hi", "boot_lo", "boot_hi"):
        rt[c] = np.exp(rt[c])             # now a geometric-mean runtime ratio NLMS/LMS
        rt["metric"] = "runtime_ratio_nlms_over_lms" 
    div = df.groupby(["condition", "method"], sort=False).diverged.sum().unstack()

    main_t.to_csv(os.path.join(R, "paired_main_snr.csv"), index=False)
    abl_t.to_csv(os.path.join(R, "paired_ablation_snr.csv"), index=False)
    qrs_t.to_csv(os.path.join(R, "paired_qrs_error.csv"), index=False)
    rt.to_csv(os.path.join(R, "runtime_ratio.csv"), index=False)
    div.to_csv(os.path.join(R, "divergence_counts.csv"))

    fmt = dict(float_format="%.3f")
    print("\nMAIN: SNR improvement, NLMS - LMS (dB), tuned at scale 1\n", main_t.to_string(index=False, **fmt))
    print("\nABLATION: NLMS - LMS (dB), both retuned per condition\n", abl_t.to_string(index=False, **fmt))
    print("\nR-peak error, NLMS - LMS (lower is better)\n", qrs_t.to_string(index=False, **fmt))
    print("\nRuntime ratio NLMS/LMS (geometric mean)\n", rt.to_string(index=False, **fmt))
    print("\nDivergence counts (test recordings)\n", div.to_string())

    forest(main_t, abl_t, os.path.join(figdir, "fig_overall_paired.png"))
    example_figure(cfg, tuned, os.path.join(figdir, "fig_example.png"),
                   ["scale1", "scale2"])
    example_figure(cfg, tuned, os.path.join(figdir, "fig_stress_example.png"),
                   ["stress_scale10", "leakage"])
    stress_figure(df, os.path.join(figdir, "fig_stress_summary.png"))
    print(f"\nfigures written to {figdir}")


if __name__ == "__main__":
    main()