import numpy as np

from endoexplain.video.overlay_renderer import RenderConfig, _apply_heatmap, _component_filter


def _cfg(**overrides):
    values = {
        "video_path": "in.avi",
        "output_path": "out.mp4",
        "classifier_ckpt": "classifier.pt",
    }
    values.update(overrides)
    return RenderConfig(**values)


def test_component_filter_rejects_excessively_large_mask(tmp_path):
    mask = np.ones((20, 20), dtype=np.uint8)
    cfg = _cfg(output_path=tmp_path / "out.mp4", mask_max_area_ratio=0.25)

    filtered, info = _component_filter(mask, cfg)

    assert int(filtered.sum()) == 0
    assert info["reason"] == "mask_area_too_large"


def test_apply_heatmap_can_render_without_a_mask(tmp_path):
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    heatmap = np.zeros((16, 16), dtype=np.float32)
    heatmap[4:12, 4:12] = 1.0
    cfg = _cfg(output_path=tmp_path / "out.mp4", heatmap_alpha=0.5, heatmap_top_percent=0.25)

    out = _apply_heatmap(frame, heatmap, cfg, mask=None)

    assert out.shape == frame.shape
    assert int(out.sum()) > 0
