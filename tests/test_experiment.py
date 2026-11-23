import numpy as np
import pandas as pd
from run_experiments import run

TINY = {
    "name": "tiny", "seconds": 4, "warmup_s": 1.0,
    "M_grid": [4], "lms_mu_grid": {"lo": 1e-2, "hi": 5e-2, "n": 2},
    "nlms_mu_grid": {"lo": 0.05, "hi": 0.5, "n": 2}, "nlms_eps": 1e-6,
    "val_seeds": [100, 101], "test_seeds": [1000, 1002], "tune_scale": 1.0,
    "timing_repeats": 1,
    "conditions": [{"name": "scale1", "scale": 1.0}, {"name": "scale10", "scale": 10.0}],
}


def test_run_produces_paired_csv(tmp_path):
    run(TINY, str(tmp_path), log=lambda *_: None)
    df = pd.read_csv(tmp_path / "results.csv")
    assert set(df.method) == {"lms", "nlms", "lms_retuned", "nlms_retuned"}
    assert len(df) == 2 * 2 * 4                      # conditions x seeds x methods
    assert df.groupby(["condition", "seed"]).size().eq(4).all()   # every case paired
    assert (df.runtime_s > 0).all()
    assert set(df.seed) == {1000, 1001}              # test seeds only, no overlap with val
    assert (tmp_path / "tuned.json").exists() and (tmp_path / "environment.json").exists()


def test_tuning_never_picks_diverging_setting():
    from src.experiment import tune
    cfg = dict(TINY)
    cfg["lms_mu_grid"] = {"lo": 1e-2, "hi": 5.0, "n": 3}   # includes a divergent mu
    best = tune("lms", cfg, range(100, 102), {"scale": 1.0})
    assert best["val_score"] > -1e8