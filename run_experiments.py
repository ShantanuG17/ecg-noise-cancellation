"""Run the full protocol: data -> tune on validation -> paired test grid -> CSV.

Usage:  python run_experiments.py --config configs/smoke.json
Outputs (created automatically): results/<name>/results.csv, tuned.json, environment.json
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import pandas as pd
import scipy

from src.experiment import run_grid


def environment():
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "numpy": np.__version__,
            "scipy": scipy.__version__, "pandas": pd.__version__}


def run(cfg, out_dir, log=print):
    os.makedirs(out_dir, exist_ok=True)
    rows, tuned = run_grid(cfg, log=log)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "results.csv"), index=False)
    with open(os.path.join(out_dir, "tuned.json"), "w") as f:
        json.dump(tuned, f, indent=2)
    with open(os.path.join(out_dir, "environment.json"), "w") as f:
        json.dump(environment(), f, indent=2)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    log(f"wrote {len(rows)} rows to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="default: results/<config name>")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = json.load(f)
    run(cfg, a.out or os.path.join("results", cfg["name"]))