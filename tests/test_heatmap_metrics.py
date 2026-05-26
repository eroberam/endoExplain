import numpy as np

from endoexplain.explainability.heatmap_metrics import explanation_metrics, threshold_heatmap


def test_top_percent_heatmap_metrics_on_simple_mask():
    heatmap = np.array([[0.0, 0.1], [0.5, 1.0]], dtype=np.float32)
    mask = np.array([[0, 0], [0, 1]], dtype=np.uint8)

    thresholded = threshold_heatmap(heatmap, mode="top_percent", value=0.25)
    metrics = explanation_metrics(heatmap, mask, threshold_mode="top_percent", threshold_value=0.25)

    assert thresholded.tolist() == [[False, False], [False, True]]
    assert metrics["explanation_iou"] == 1.0
    assert metrics["pointing_game_hit"] is True
    assert 0.62 < metrics["activation_inside_mask"] < 0.63
