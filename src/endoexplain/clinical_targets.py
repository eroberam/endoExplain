"""Clinical target helpers used by video review and metric exports.

The dataset keeps ``polyps`` and ``dyed-lifted-polyps`` as separate labels.
For clinical review they are usually handled as one positive family because
both indicate a polyp lesion, with the dye status describing the acquisition
context rather than a different review target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


POLYP_FAMILY: tuple[str, ...] = ("polyps", "dyed-lifted-polyps")


@dataclass(frozen=True)
class ClinicalTarget:
    """Resolved positive target used for scoring and visual review."""

    display_name: str
    class_labels: tuple[str, ...]
    class_indices: tuple[int, ...]


def pretty_label(label: str | None) -> str:
    """Return a readable label without changing the underlying taxonomy."""
    if not label:
        return ""
    if label == "polyp_family":
        return "polyp family"
    return str(label).replace("_", " ").replace("-", " ")


def resolve_target(
    class_to_idx: dict[str, int],
    suspicious_class: str | None = None,
    positive_classes: Sequence[str] | None = None,
    target_display_name: str | None = None,
) -> ClinicalTarget:
    """Resolve a visual-review target from dataset labels.

    If ``suspicious_class`` is ``polyps`` and the classifier also contains
    ``dyed-lifted-polyps``, both labels are summed into a single polyp-family
    score. Explicit ``positive_classes`` always wins.
    """
    if positive_classes:
        labels = tuple(c for c in positive_classes if c in class_to_idx)
        display = target_display_name or pretty_label("+".join(labels))
        return ClinicalTarget(display, labels, tuple(class_to_idx[c] for c in labels))

    if suspicious_class in {"polyp_family", "polyps"}:
        labels = tuple(c for c in POLYP_FAMILY if c in class_to_idx)
        if labels:
            return ClinicalTarget(
                target_display_name or "polyp family",
                labels,
                tuple(class_to_idx[c] for c in labels),
            )

    if suspicious_class and suspicious_class in class_to_idx:
        return ClinicalTarget(
            target_display_name or pretty_label(suspicious_class),
            (suspicious_class,),
            (class_to_idx[suspicious_class],),
        )

    return ClinicalTarget(target_display_name or "predicted class", (), ())


def target_probability(probs: np.ndarray, target: ClinicalTarget, pred_idx: int) -> float:
    """Probability used by temporal review.

    For clinical families the probabilities of all positive labels are summed.
    If no explicit target is configured, the predicted-class probability is used.
    """
    if target.class_indices:
        return float(np.asarray(probs)[list(target.class_indices)].sum())
    return float(np.asarray(probs)[int(pred_idx)])


def cam_target_class(
    class_to_idx: dict[str, int],
    target: ClinicalTarget,
    pred_label: str,
) -> int | None:
    """Pick the classifier class used for class-discriminative Grad-CAM."""
    if "polyps" in target.class_labels:
        return class_to_idx["polyps"]
    if target.class_labels:
        return class_to_idx[target.class_labels[0]]
    return class_to_idx.get(pred_label)


def binary_label_for_target(label: str, positives: Iterable[str]) -> int:
    """Return 1 when ``label`` belongs to ``positives``."""
    return int(label in set(positives))
