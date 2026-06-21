from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def inverse_frequency_weights(class_counts: torch.Tensor) -> torch.Tensor:
    counts = class_counts.float().clamp_min(1.0)
    weights = counts.sum() / (len(counts) * counts)
    return weights / weights.mean().clamp_min(1e-12)


class CrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing: float = 0.0, weights: torch.Tensor | None = None):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.register_buffer("weights", None if weights is None else weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.register_buffer("weights", None if weights is None else weights.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        target_probability = logits.softmax(dim=1).gather(1, targets[:, None]).squeeze(1)
        loss = (1.0 - target_probability).pow(self.gamma) * ce
        if self.weights is not None:
            loss = loss * self.weights[targets]
        return loss.mean()


class PriorAdjustedCrossEntropy(nn.Module):
    def __init__(self, class_counts: torch.Tensor, tau: float, label_smoothing: float):
        super().__init__()
        prior = class_counts.float().clamp_min(1.0)
        prior = prior / prior.sum()
        self.register_buffer("adjustment", tau * prior.log())
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits + self.adjustment,
            targets,
            label_smoothing=self.label_smoothing,
        )


@dataclass(frozen=True)
class ClassificationLossConfig:
    name: str
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    focal_use_class_weights: bool = False
    logit_adjustment_tau: float = 1.0


def build_classification_loss(
    cfg: ClassificationLossConfig,
    class_counts: torch.Tensor,
) -> nn.Module:
    if cfg.name == "ce":
        return CrossEntropyLoss(cfg.label_smoothing)
    if cfg.name == "weighted_ce":
        return CrossEntropyLoss(cfg.label_smoothing, inverse_frequency_weights(class_counts))
    if cfg.name == "focal":
        weights = inverse_frequency_weights(class_counts) if cfg.focal_use_class_weights else None
        return FocalLoss(cfg.focal_gamma, cfg.label_smoothing, weights)
    if cfg.name == "logit_adjustment":
        return PriorAdjustedCrossEntropy(class_counts, cfg.logit_adjustment_tau, cfg.label_smoothing)
    if cfg.name == "balanced_softmax":
        return PriorAdjustedCrossEntropy(class_counts, 1.0, cfg.label_smoothing)
    raise ValueError(f"Unknown classification loss: {cfg.name}")
