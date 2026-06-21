from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ClassifierMode = Literal["linear", "arcface", "cosface"]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    classification_loss: str = "ce"
    metric_loss: str | None = None
    two_views: bool = False
    classifier_mode: ClassifierMode = "linear"
    approximate: bool = False

    @property
    def classification_only(self) -> bool:
        return self.metric_loss is None


METHODS: dict[str, MethodSpec] = {
    "ce": MethodSpec("ce"),
    "weighted_ce": MethodSpec("weighted_ce", classification_loss="weighted_ce"),
    "focal": MethodSpec("focal", classification_loss="focal"),
    "logit_adjustment": MethodSpec("logit_adjustment", classification_loss="logit_adjustment"),
    "balanced_softmax": MethodSpec("balanced_softmax", classification_loss="balanced_softmax"),
    "supcon": MethodSpec("supcon", metric_loss="supcon", two_views=True),
    "triplet": MethodSpec("triplet", metric_loss="triplet"),
    "n_pairs": MethodSpec("n_pairs", metric_loss="n_pairs", two_views=True),
    "multi_similarity": MethodSpec("multi_similarity", metric_loss="multi_similarity"),
    "circle": MethodSpec("circle", metric_loss="circle"),
    "proxy_anchor": MethodSpec("proxy_anchor", metric_loss="proxy_anchor"),
    "center": MethodSpec("center", metric_loss="center"),
    "prototype": MethodSpec("prototype", metric_loss="prototype"),
    "meta_prototype": MethodSpec("meta_prototype", metric_loss="prototype", approximate=True),
    "arcface": MethodSpec("arcface", classifier_mode="arcface"),
    "cosface": MethodSpec("cosface", classifier_mode="cosface"),
    "paco_lite": MethodSpec("paco_lite", metric_loss="paco_lite", two_views=True, approximate=True),
    "bcl_lite": MethodSpec("bcl_lite", metric_loss="bcl_lite", two_views=True, approximate=True),
    "sbcl_lite": MethodSpec("sbcl_lite", metric_loss="sbcl_lite", two_views=True, approximate=True),
}


def get_method(name: str) -> MethodSpec:
    try:
        return METHODS[name]
    except KeyError as error:
        raise ValueError(f"Unknown method {name!r}; choose {sorted(METHODS)}") from error


def all_methods() -> list[str]:
    return sorted(METHODS)


def needs_two_views(name: str) -> bool:
    return get_method(name).two_views

