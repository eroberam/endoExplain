import pandas as pd

from endoexplain.evaluation.temporal_benchmark import (
    evaluate_temporal_benchmark,
    events_from_frame_scores,
    match_temporal_events,
    validate_temporal_inputs,
)


def test_events_from_frame_scores_groups_by_gap():
    frames = pd.DataFrame(
        {
            "timestamp": [0.0, 0.2, 0.4, 2.0, 2.2],
            "score": [0.9, 0.8, 0.1, 0.95, 0.96],
        }
    )

    events = events_from_frame_scores(
        frames,
        score_column="score",
        threshold=0.5,
        max_gap_seconds=0.5,
    )

    assert len(events) == 2
    assert events.iloc[0]["start_time"] == 0.0
    assert events.iloc[1]["start_time"] == 2.0


def test_match_temporal_events_is_one_to_one_and_reports_latency():
    truth = pd.DataFrame(
        {
            "video_id": ["v1", "v1"],
            "event_id": [1, 2],
            "start_s": [1.0, 5.0],
            "end_s": [3.0, 6.0],
        }
    )
    pred = pd.DataFrame(
        {
            "event_id": [1, 2],
            "start_time": [1.5, 8.0],
            "end_time": [2.5, 9.0],
        }
    )

    matches, fps, fns = match_temporal_events(truth, pred)

    assert len(matches) == 1
    assert matches[0].latency_seconds == 0.5
    assert fps == [1]
    assert fns == [1]


def test_temporal_benchmark_summary_counts_false_events_and_spike_reduction():
    videos = pd.DataFrame(
        {
            "video_id": ["v1"],
            "relative_path": ["v1.avi"],
            "source_label": ["polyps"],
            "target_binary": [1],
            "fps": [5.0],
            "frame_count": [20],
            "duration_seconds": [4.0],
            "role": ["positive_polyps"],
        }
    )
    truth = pd.DataFrame({"video_id": ["v1"], "event_id": [1], "start_s": [1.0], "end_s": [2.5]})
    frames = pd.DataFrame(
        {
            "timestamp": [0.0, 0.2, 1.0, 1.2, 1.4, 3.4, 3.6],
            "target_probability": [0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
            "target_probability_smoothed": [0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1],
        }
    )

    per_video, summary = evaluate_temporal_benchmark(
        videos,
        truth,
        {"v1": frames},
        threshold=0.5,
        max_gap_seconds=0.5,
    )

    assert per_video.iloc[0]["tp"] == 1
    assert summary["event_recall"] == 1.0
    assert summary["spike_reduction"] > 0.0


def test_temporal_csv_schema_validation():
    videos = pd.DataFrame(
        {
            "video_id": ["v1"],
            "relative_path": ["v1.avi"],
            "source_label": ["polyps"],
            "target_binary": [1],
            "fps": [25.0],
            "frame_count": [100],
            "duration_seconds": [4.0],
            "role": ["positive_polyps"],
        }
    )
    events = pd.DataFrame({"video_id": ["v1"], "event_id": [1], "start_s": [0.0], "end_s": [1.0]})

    assert validate_temporal_inputs(videos, events) == []
