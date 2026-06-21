from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def pair_masks(targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    same = targets[:, None].eq(targets[None, :])
    eye = torch.eye(len(targets), device=targets.device, dtype=torch.bool)
    return same & ~eye, ~same


def normalized(proxies: torch.Tensor | None) -> torch.Tensor:
    if proxies is None:
        raise ValueError("This metric loss requires learnable class proxies")
    return F.normalize(proxies, dim=1)


def multi_positive_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    denominator_mask: torch.Tensor,
) -> torch.Tensor:
    positive_mask = positive_mask & denominator_mask
    valid = positive_mask.any(dim=1)
    if not valid.any():
        return logits.sum() * 0.0
    masked = logits.masked_fill(~denominator_mask, -torch.inf)
    log_prob = logits - torch.logsumexp(masked, dim=1, keepdim=True)
    positives = positive_mask.float()
    loss = -(log_prob.masked_fill(~positive_mask, 0.0) * positives).sum(1)
    loss = loss / positives.sum(1).clamp_min(1.0)
    return loss[valid].mean()


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        logits = embeddings @ embeddings.T / self.temperature
        positives, _ = pair_masks(targets)
        denominator = ~torch.eye(len(targets), device=targets.device, dtype=torch.bool)
        return multi_positive_loss(logits, positives, denominator)


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        distances = torch.cdist(embeddings.float(), embeddings.float())
        positives, negatives = pair_masks(targets)
        valid = positives.any(1) & negatives.any(1)
        if not valid.any():
            return embeddings.sum() * 0.0
        hardest_positive = distances.masked_fill(~positives, -torch.inf).max(1).values
        hardest_negative = distances.masked_fill(~negatives, torch.inf).min(1).values
        return F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin).mean()


class MultiSimilarityLoss(nn.Module):
    def __init__(self, margin: float = 0.5, alpha: float = 2.0, beta: float = 50.0):
        super().__init__()
        self.margin, self.alpha, self.beta = margin, alpha, beta

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        similarities = embeddings @ embeddings.T
        positives, negatives = pair_masks(targets)
        pos = torch.exp(-self.alpha * (similarities - self.margin)).masked_fill(~positives, 0).sum(1)
        neg = torch.exp(self.beta * (similarities - self.margin)).masked_fill(~negatives, 0).sum(1)
        return (torch.log1p(pos) / self.alpha + torch.log1p(neg) / self.beta).mean()


class CircleLoss(nn.Module):
    def __init__(self, margin: float = 0.25, gamma: float = 64.0):
        super().__init__()
        self.margin, self.gamma = margin, gamma

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, **_) -> torch.Tensor:
        similarities = embeddings @ embeddings.T
        positives, negatives = pair_masks(targets)
        alpha_p = (-similarities.detach() + 1 + self.margin).clamp_min(0)
        alpha_n = (similarities.detach() + self.margin).clamp_min(0)
        pos = (-self.gamma * alpha_p * (similarities - (1 - self.margin))).masked_fill(~positives, -torch.inf)
        neg = (self.gamma * alpha_n * (similarities - self.margin)).masked_fill(~negatives, -torch.inf)
        valid = positives.any(1) & negatives.any(1)
        if not valid.any():
            return embeddings.sum() * 0.0
        return F.softplus(torch.logsumexp(pos[valid], 1) + torch.logsumexp(neg[valid], 1)).mean()


class ProxyAnchorLoss(nn.Module):
    def __init__(self, margin: float = 0.1, alpha: float = 32.0):
        super().__init__()
        self.margin, self.alpha = margin, alpha

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, proxies=None, **_) -> torch.Tensor:
        proxies = normalized(proxies)
        similarities = embeddings @ proxies.T
        one_hot = F.one_hot(targets, len(proxies)).bool()
        pos_values = (-self.alpha * (similarities - self.margin)).T.masked_fill(~one_hot.T, -torch.inf)
        neg_values = (self.alpha * (similarities + self.margin)).T.masked_fill(one_hot.T, -torch.inf)
        zero = torch.zeros((len(proxies), 1), device=embeddings.device, dtype=embeddings.dtype)
        positive = torch.logsumexp(torch.cat([zero, pos_values], 1), 1)
        negative = torch.logsumexp(torch.cat([zero, neg_values], 1), 1)
        present = one_hot.any(0)
        return positive[present].mean() + negative.mean()


class CenterLoss(nn.Module):
    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, proxies=None, **_) -> torch.Tensor:
        centers = normalized(proxies)
        return (embeddings - centers[targets]).square().sum(1).mean()


class PrototypeLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, proxies=None, **_) -> torch.Tensor:
        return F.cross_entropy(embeddings @ normalized(proxies).T / self.temperature, targets)


class PaCoLiteLoss(nn.Module):
    """Compact PaCo-inspired loss; not an exact reproduction of the paper."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor, proxies=None, **_) -> torch.Tensor:
        centers = normalized(proxies)
        keys = torch.cat([centers, embeddings])
        key_targets = torch.cat([torch.arange(len(centers), device=targets.device), targets])
        logits = embeddings @ keys.T / self.temperature
        positives = targets[:, None].eq(key_targets[None])
        denominator = torch.ones_like(positives)
        indices = len(centers) + torch.arange(len(embeddings), device=targets.device)
        denominator[torch.arange(len(embeddings), device=targets.device), indices] = False
        positives &= denominator
        return multi_positive_loss(logits, positives, denominator)


class BCLLiteLoss(PaCoLiteLoss):
    """BCL-inspired class-balanced denominator using global train counts."""

    def forward(self, embeddings, targets, proxies=None, class_counts=None, **_):
        if class_counts is None:
            raise ValueError("bcl_lite requires global train class_counts")
        centers = normalized(proxies)
        key_targets = torch.cat([torch.arange(len(centers), device=targets.device), targets])
        keys = torch.cat([centers, embeddings])
        logits = embeddings @ keys.T / self.temperature
        logits = logits - class_counts.to(logits.device).float().clamp_min(1).log()[key_targets]
        positives = targets[:, None].eq(key_targets[None])
        denominator = torch.ones_like(positives)
        indices = len(centers) + torch.arange(len(embeddings), device=targets.device)
        denominator[torch.arange(len(embeddings), device=targets.device), indices] = False
        positives &= denominator
        return multi_positive_loss(logits, positives, denominator)


class SBCLLiteLoss(BCLLiteLoss):
    """Class-balanced approximation only; it does not claim official SBCL subclass mining."""


def build_metric_loss(name: str | None, temperature: float = 0.1, margin: float = 0.2) -> nn.Module | None:
    if name is None:
        return None
    table: dict[str, nn.Module] = {
        "supcon": SupConLoss(temperature),
        "n_pairs": SupConLoss(temperature),
        "triplet": BatchHardTripletLoss(margin),
        "multi_similarity": MultiSimilarityLoss(),
        "circle": CircleLoss(),
        "proxy_anchor": ProxyAnchorLoss(margin),
        "center": CenterLoss(),
        "prototype": PrototypeLoss(temperature),
        "paco_lite": PaCoLiteLoss(temperature),
        "bcl_lite": BCLLiteLoss(temperature),
        "sbcl_lite": SBCLLiteLoss(temperature),
    }
    try:
        return table[name]
    except KeyError as error:
        raise ValueError(f"Unknown metric loss: {name}") from error
