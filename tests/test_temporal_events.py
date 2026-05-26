import pandas as pd

from endoexplain.temporal.event_grouping import EventConfig, group_events


def test_group_events_uses_smoothed_confidence_and_gap_threshold():
    frames = pd.DataFrame(
        {
            "timestamp": [0.0, 0.1, 0.2, 1.5, 1.6],
            "confidence": [0.2, 0.2, 0.2, 0.2, 0.2],
            "confidence_smoothed": [0.90, 0.80, 0.10, 0.92, 0.91],
        }
    )

    events = group_events(
        frames,
        EventConfig(confidence_threshold=0.5, max_gap_seconds=0.3),
    )

    assert len(events) == 2
    assert events.iloc[0]["start_time"] == 0.0
    assert events.iloc[0]["end_time"] == 0.1
    assert events.iloc[1]["start_time"] == 1.5
