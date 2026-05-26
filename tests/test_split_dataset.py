import random

import pandas as pd

from endoexplain.data.split_dataset import group_split_by_video


def test_group_split_by_video_is_seeded_and_keeps_video_frames_together():
    rows = [
        {"video_id": f"video_{video_idx:02d}", "frame": frame_idx}
        for video_idx in range(10)
        for frame_idx in range(3)
    ]
    df = pd.DataFrame(rows)

    out = group_split_by_video(df, "video_id", fractions=(0.5, 0.2, 0.3), seed=7)

    per_video = out.groupby("video_id")["split"].nunique()
    assert int(per_video.max()) == 1

    expected = sorted(df["video_id"].unique().tolist())
    random.Random(7).shuffle(expected)
    expected_train = set(expected[:5])
    expected_val = set(expected[5:7])

    mapping = out.groupby("video_id")["split"].first().to_dict()
    assert {v for v, split in mapping.items() if split == "train"} == expected_train
    assert {v for v, split in mapping.items() if split == "val"} == expected_val
