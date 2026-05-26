import pandas as pd

from endoexplain.evaluation.bootstrap import bootstrap_metric


def test_bootstrap_metric_is_deterministic():
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})

    a = bootstrap_metric(data, lambda df: df["x"].mean(), n_boot=200, seed=45)
    b = bootstrap_metric(data, lambda df: df["x"].mean(), n_boot=200, seed=45)

    assert a == b
    assert a["estimate"] == 2.5
    assert a["ci_low"] <= a["estimate"] <= a["ci_high"]


def test_bootstrap_metric_accepts_strata():
    data = pd.DataFrame({"y": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})

    out = bootstrap_metric(
        data,
        lambda df: df["score"].mean(),
        n_boot=100,
        seed=45,
        strata=data["y"],
    )

    assert out["n"] == 4
    assert out["ci_low"] <= out["estimate"] <= out["ci_high"]
