from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

import timm


MarginMode = Literal["arcface", "cosface"]


class AngularMarginHead(nn.Module):
    """
    Classification head for ArcFace / CosFace.

    It operates on L2-normalized embeddings and normalized class weights.

    During training:
      - labels are provided;
      - target-class logits receive angular/cosine margin.

    During evaluation:
      - labels are None;
      - returns regular scaled cosine logits.
    """

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        scale: float = 30.0,
        margin: float = 0.2,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(
            torch.empty(n_classes, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        mode: MarginMode = "arcface",
        margin: float | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        margin = self.margin if margin is None else margin
        scale = self.scale if scale is None else scale

        embeddings = F.normalize(embeddings, dim=1)
        weights = F.normalize(self.weight, dim=1)

        cosine = F.linear(embeddings, weights)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Inference mode: no target margin.
        if labels is None:
            return cosine * scale

        if mode == "arcface":
            target_logits = self._arcface_target(cosine, margin)
        elif mode == "cosface":
            target_logits = cosine - margin
        else:
            raise ValueError(
                f"Unknown margin mode: {mode}. "
                "Expected 'arcface' or 'cosface'."
            )

        one_hot = F.one_hot(
            labels,
            num_classes=self.n_classes,
        ).to(dtype=torch.bool, device=cosine.device)

        logits = torch.where(one_hot, target_logits, cosine)
        logits = logits * scale

        return logits

    @staticmethod
    def _arcface_target(
        cosine: torch.Tensor,
        margin: float,
    ) -> torch.Tensor:
        """
        Computes cos(theta + margin) in a numerically stable form:
        cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        """
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(1e-7))

        cos_m = math.cos(margin)
        sin_m = math.sin(margin)

        phi = cosine * cos_m - sine * sin_m
        threshold = math.cos(math.pi - margin)
        correction = math.sin(math.pi - margin) * margin
        return torch.where(cosine > threshold, phi, cosine - correction)


class ConvNeXtMetric(nn.Module):
    """
    ConvNeXt backbone with separate heads:

    1. classifier(features) -> standard logits
       Used for CE / Weighted CE / Focal / Logit Adjustment / Balanced Softmax.

    2. projector(features) -> normalized embedding z
       Used for SupCon / Triplet / ProxyAnchor / Center / BCL / PaCo-like losses.

    3. angular_classifier(z, labels) -> ArcFace/CosFace logits
       Used for angular-margin classification methods.
    """

    def __init__(
        self,
        name: str,
        n_classes: int = 7,
        embedding_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
        angular_scale: float = 30.0,
        angular_margin: float = 0.2,
    ):
        super().__init__()

        self.name = name
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, n_classes),
        )

        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, embedding_dim),
        )

        self.angular_classifier = AngularMarginHead(
            embedding_dim=embedding_dim,
            n_classes=n_classes,
            scale=angular_scale,
            margin=angular_margin,
        )

        # Learnable class proxies / centers in embedding space.
        # Used by ProxyAnchor, CenterLoss, PrototypeContrastive, PaCo-lite, etc.
        self.proxies = nn.Parameter(
            torch.empty(n_classes, embedding_dim)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.proxies, mean=0.0, std=0.02)

        last = self.classifier[-1]

        if isinstance(last, nn.Linear):
            nn.init.trunc_normal_(last.weight, std=0.02)
            nn.init.zeros_(last.bias)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns backbone features before classification/projection heads.
        Shape: [batch, feature_dim].
        """
        return self.backbone(x)

    def embed_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized embedding for metric-learning objectives.
        Shape: [batch, embedding_dim].
        """
        z = self.projector(features)
        return F.normalize(z, dim=1)

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns standard classification logits from raw backbone features.
        Shape: [batch, n_classes].
        """
        return self.classifier(features)

    def classify_embeddings_with_margin(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        mode: MarginMode = "arcface",
        margin: float | None = None,
        scale: float | None = None,
    ) -> torch.Tensor:
        """
        Returns ArcFace/CosFace logits from normalized embedding space.

        labels:
          - provided during training to apply target-class margin;
          - None during validation/test to use plain scaled cosine logits.
        """
        return self.angular_classifier(
            embeddings,
            labels=labels,
            mode=mode,
            margin=margin,
            scale=scale,
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        classifier_mode: Literal["linear", "arcface", "cosface"] = "linear",
        compute_embedding: bool = True,
        embedding_source: Literal["projector", "backbone"] = "projector",
        margin: float | None = None,
        scale: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.extract_features(x)
        embeddings = None

        if compute_embedding or classifier_mode in {"arcface", "cosface"}:
            embeddings = (
                self.embed_features(features)
                if embedding_source == "projector" or classifier_mode in {"arcface", "cosface"}
                else F.normalize(features, dim=1)
            )

        if classifier_mode == "linear":
            logits = self.classify_features(features)
        elif classifier_mode in {"arcface", "cosface"}:
            if embeddings is None:
                raise RuntimeError("Angular-margin classification requires embeddings")
            logits = self.classify_embeddings_with_margin(
                embeddings,
                labels=labels,
                mode=classifier_mode,
                margin=margin,
                scale=scale,
            )
        else:
            raise ValueError(
                f"Unknown classifier_mode={classifier_mode!r}. "
                "Expected 'linear', 'arcface' or 'cosface'."
            )

        return logits, embeddings

    def set_checkpointing(self, enabled: bool = True) -> None:
        """
        Enables gradient checkpointing for timm backbones that support it.
        Useful when VRAM is insufficient.
        """
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enabled)
