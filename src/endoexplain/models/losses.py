"""Losses for binary polyp segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for single-channel binary segmentation.

    Expects logits of shape (B, 1, H, W) and targets in {0, 1} of shape
    (B, 1, H, W).
    """

    def __init__(self, eps: float = 1.0) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE-with-logits + Dice (1:1 by default)."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(
            logits, targets
        )


class FocalLoss(nn.Module):
    """Binary focal loss for imbalanced foreground/background pixels."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt).pow(self.gamma) * bce).mean()


class TverskyLoss(nn.Module):
    """Tversky loss; beta > alpha favours recall."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, eps: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        score = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1.0 - score.mean()


class CompositeSegLoss(nn.Module):
    """Weighted BCE + Dice + optional focal/Tversky loss."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        tversky_weight: float = 0.0,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
    ) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.bce_weight * self.bce(logits, targets)
        loss = loss + self.dice_weight * self.dice(logits, targets)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        if self.tversky_weight > 0:
            loss = loss + self.tversky_weight * self.tversky(logits, targets)
        return loss


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Hard-thresholded Dice for monitoring."""
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (1, 2, 3)
    inter = (preds * targets).sum(dim=dims)
    denom = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    return float(dice.mean().item())


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (1, 2, 3)
    inter = (preds * targets).sum(dim=dims)
    union = ((preds + targets) > 0).float().sum(dim=dims)
    iou = (inter + 1.0) / (union + 1.0)
    return float(iou.mean().item())


@torch.no_grad()
def binary_segmentation_stats(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)
    inter = tp
    pred_sum = preds.sum(dim=dims)
    target_sum = targets.sum(dim=dims)
    union = ((preds + targets) > 0).float().sum(dim=dims)
    dice = (2.0 * inter + 1.0) / (pred_sum + target_sum + 1.0)
    iou = (inter + 1.0) / (union + 1.0)
    precision = (tp + 1.0) / (tp + fp + 1.0)
    recall = (tp + 1.0) / (tp + fn + 1.0)
    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }
