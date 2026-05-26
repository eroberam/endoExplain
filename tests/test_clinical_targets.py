import numpy as np

from endoexplain.clinical_targets import cam_target_class, resolve_target, target_probability


def test_polyp_family_sums_polyps_and_dyed_lifted_polyps():
    class_to_idx = {"dyed-lifted-polyps": 0, "polyps": 1, "cecum": 2}
    target = resolve_target(class_to_idx, suspicious_class="polyps")
    probs = np.array([0.25, 0.55, 0.20], dtype=np.float32)

    assert target.display_name == "polyp family"
    assert target.class_labels == ("polyps", "dyed-lifted-polyps")
    assert target_probability(probs, target, pred_idx=2) == float(probs[[1, 0]].sum())
    assert cam_target_class(class_to_idx, target, pred_label="cecum") == 1
