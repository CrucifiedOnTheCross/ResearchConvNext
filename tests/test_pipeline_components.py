from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ham_pipeline.classification_losses import ClassificationLossConfig, build_classification_loss
from ham_pipeline.config import load_config
from ham_pipeline.losses import build_metric_loss
from ham_pipeline.methods import get_method
from ham_pipeline.metrics import evaluate_arrays
from ham_pipeline.model import AngularMarginHead


class PipelineComponentTests(unittest.TestCase):
    def test_method_contracts(self):
        self.assertTrue(get_method("supcon").two_views)
        self.assertFalse(get_method("ce").two_views)
        self.assertIsNone(get_method("arcface").metric_loss)
        self.assertEqual(get_method("arcface").classifier_mode, "arcface")
        self.assertTrue(get_method("bcl_lite").approximate)

    def test_config_inheritance(self):
        cfg = load_config("configs/focal.yaml")
        self.assertEqual(cfg["training"]["method"], "focal")
        self.assertEqual(cfg["model"]["name"], "convnext_base.fb_in22k_ft_in1k")
        self.assertEqual(cfg["training"]["metric_weight"], 0.0)

    def test_classification_losses_are_finite(self):
        logits = torch.randn(12, 7, requires_grad=True)
        targets = torch.arange(12) % 7
        counts = torch.tensor([10, 20, 30, 40, 50, 60, 70])
        for name in ("ce", "weighted_ce", "focal", "logit_adjustment", "balanced_softmax"):
            loss = build_classification_loss(ClassificationLossConfig(name=name), counts)(logits, targets)
            self.assertTrue(torch.isfinite(loss), name)

    def test_arcface_head(self):
        head = AngularMarginHead(16, 7)
        embeddings = torch.randn(8, 16)
        targets = torch.arange(8) % 7
        train_logits = head(embeddings, targets, mode="arcface")
        eval_logits = head(embeddings)
        self.assertEqual(train_logits.shape, (8, 7))
        self.assertFalse(torch.allclose(train_logits, eval_logits))

    def test_bcl_uses_global_counts(self):
        embeddings = torch.nn.functional.normalize(torch.randn(14, 16), dim=1)
        targets = torch.arange(14) % 7
        proxies = torch.randn(7, 16, requires_grad=True)
        loss_fn = build_metric_loss("bcl_lite")
        balanced = loss_fn(embeddings, targets, proxies=proxies, class_counts=torch.ones(7))
        imbalanced = loss_fn(
            embeddings,
            targets,
            proxies=proxies,
            class_counts=torch.tensor([100, 50, 20, 10, 5, 2, 1]),
        )
        self.assertFalse(torch.allclose(balanced, imbalanced))

    def test_validation_thresholds_are_reused_on_test(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(size=(70, 7)).astype(np.float32)
        labels = np.tile(np.arange(7), 10)
        embeddings = rng.normal(size=(70, 8)).astype(np.float32)
        val, _, _, temperature = evaluate_arrays(
            logits,
            labels,
            embeddings,
            ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            fit_temperature=True,
            fit_thresholds_on_this_split=True,
        )
        test, _, _, _ = evaluate_arrays(
            logits,
            labels,
            embeddings,
            ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
            temperature=temperature,
            threshold_state=val["threshold_state"],
            fit_thresholds_on_this_split=False,
        )
        self.assertEqual(val["threshold_state"], test["threshold_state"])


if __name__ == "__main__":
    unittest.main()
