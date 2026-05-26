import pandas as pd

from endoexplain.evaluation.quality_strata import add_quality_strata, summarize_quality_strata


def test_quality_strata_are_mutually_exclusive():
    df = pd.DataFrame(
        [
            {"blur_flag": False, "overexposed_frame_flag": False, "reflection_flag": False},
            {"blur_flag": True, "overexposed_frame_flag": False, "reflection_flag": False},
            {"blur_flag": False, "overexposed_frame_flag": True, "reflection_flag": False},
            {"blur_flag": False, "overexposed_frame_flag": False, "reflection_flag": True},
            {"blur_flag": True, "overexposed_frame_flag": False, "reflection_flag": True},
        ]
    )

    out = add_quality_strata(df)

    assert out["quality_stratum"].tolist() == [
        "good",
        "blurred",
        "overexposed",
        "reflective",
        "mixed_low_quality",
    ]


def test_quality_summary_reports_available_metrics():
    df = pd.DataFrame(
        [
            {"blur_flag": False, "overexposed_frame_flag": False, "reflection_flag": False, "correct": 1},
            {"blur_flag": False, "overexposed_frame_flag": False, "reflection_flag": False, "correct": 0},
        ]
    )

    summary = summarize_quality_strata(df, min_n=1)
    good = summary[summary["quality_stratum"] == "good"].iloc[0]

    assert good["n"] == 2
    assert good["accuracy"] == 0.5
