import numpy as np
import pandas as pd
from src.analysis import paired_ci, paired_table


def test_paired_ci_constant_differences():
    r = paired_ci([1.0, 1.0, 1.0, 1.0])
    assert r["n"] == 4 and np.isclose(r["mean"], 1.0)
    assert np.isclose(r["t_lo"], 1.0) and np.isclose(r["boot_hi"], 1.0)


def test_paired_ci_drops_nonfinite_and_reports_n():
    r = paired_ci([1.0, 3.0, np.nan, np.inf])
    assert r["n"] == 2 and np.isclose(r["mean"], 2.0)


def test_paired_table_pairs_by_seed():
    rows = []
    for seed, (x, y) in enumerate([(5.0, 3.0), (6.0, 3.0), (7.0, 3.0)]):
        rows.append(dict(seed=seed, condition="c", method="a", v=x, diverged=False))
        rows.append(dict(seed=seed, condition="c", method="b", v=y, diverged=False))
    t = paired_table(pd.DataFrame(rows), "a", "b", "v")
    assert np.isclose(t.loc[0, "mean"], 3.0) and t.loc[0, "n"] == 3