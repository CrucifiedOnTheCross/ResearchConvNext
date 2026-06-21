from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def nested_get(obj: dict, path: str, default=None):
    cur: Any = obj

    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default

        cur = cur[part]

    return cur


def to_float(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return value


def add_metric_prefix(row: dict, metrics: dict, split: str) -> None:
    """
    Adds the most important metrics from val/test_metrics.json.
    """

    prefix = f"{split}_"

    # Main multiclass metrics.
    for key in [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "roc_auc_macro_ovr",
        "pr_auc_macro",
    ]:
        row[prefix + key] = to_float(metrics.get(key))

    # Calibration before/after temperature scaling.
    for stage in ["calibration_before", "calibration_after"]:
        for key in ["ece", "brier", "nll"]:
            row[f"{prefix}{stage}_{key}"] = to_float(
                nested_get(metrics, f"{stage}.{key}")
            )

    row[prefix + "temperature"] = to_float(
        metrics.get("temperature_from_validation")
    )

    # Embedding geometry.
    for key in [
        "silhouette",
        "intra_class_distance",
        "inter_class_distance",
        "inter_intra_ratio",
    ]:
        row[prefix + "embedding_" + key] = to_float(
            nested_get(metrics, f"embedding.{key}")
        )

    # Fixed malignant metrics for backward compatibility.
    for key in [
        "threshold",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
        "balanced_accuracy",
        "mcc",
        "roc_auc",
        "pr_auc",
    ]:
        row[f"{prefix}malignant_fixed_0_5_{key}"] = to_float(
            nested_get(metrics, f"binary_malignant.{key}")
        )

    # Thresholded malignant and melanoma metrics.
    for task in ["malignant", "melanoma"]:
        for criterion in [
            "fixed_0_5",
            "youden",
            "mcc",
            "f1",
            "balanced_accuracy",
        ]:
            base = f"threshold_analysis.{task}.metrics.{criterion}"

            for key in [
                "threshold",
                "sensitivity",
                "specificity",
                "precision",
                "recall",
                "f1",
                "balanced_accuracy",
                "mcc",
                "roc_auc",
                "pr_auc",
                "tp",
                "fp",
                "tn",
                "fn",
            ]:
                row[f"{prefix}{task}_{criterion}_{key}"] = to_float(
                    nested_get(metrics, f"{base}.{key}")
                )

    # Per-class metrics.
    per_class = metrics.get("per_class", {})

    for class_name, class_metrics in per_class.items():
        safe_class = str(class_name).replace("-", "_")

        for key in ["precision", "recall", "f1-score", "support"]:
            out_key = key.replace("-", "_")
            row[f"{prefix}{safe_class}_{out_key}"] = to_float(
                class_metrics.get(key)
            )

    # Per-class AUC.
    per_class_auc = metrics.get("per_class_auc", {})

    for class_name, class_metrics in per_class_auc.items():
        safe_class = str(class_name).replace("-", "_")

        for key in ["roc_auc", "pr_auc"]:
            row[f"{prefix}{safe_class}_{key}"] = to_float(
                class_metrics.get(key)
            )


def collect_run(run_dir: Path) -> dict | None:
    config = read_yaml(run_dir / "config.yaml")
    best = read_json(run_dir / "best_summary.json")
    val_metrics = read_json(run_dir / "val_metrics.json")
    test_metrics = read_json(run_dir / "test_metrics.json")
    class_counts = read_json(run_dir / "class_counts.json")

    if not config and not test_metrics:
        return None

    row = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,

        "method": nested_get(config, "training.method"),
        "seed": nested_get(config, "seed"),

        "backbone": nested_get(config, "model.name"),
        "embedding_dim": nested_get(config, "model.embedding_dim"),
        "image_size": nested_get(config, "data.image_size"),

        "epochs_config": nested_get(config, "training.epochs"),
        "batch_size_config": nested_get(config, "training.batch_size"),
        "effective_batch_size": nested_get(config, "training.effective_batch_size"),
        "precision": nested_get(config, "training.precision"),

        "learning_rate": nested_get(config, "training.learning_rate"),
        "head_learning_rate": nested_get(config, "training.head_learning_rate"),
        "weight_decay": nested_get(config, "training.weight_decay"),

        "ce_weight": nested_get(config, "training.ce_weight"),
        "metric_weight": nested_get(config, "training.metric_weight"),
        "temperature": nested_get(config, "training.temperature"),
        "margin": nested_get(config, "training.margin"),
        "focal_gamma": nested_get(config, "training.focal_gamma"),
        "logit_adjustment_tau": nested_get(config, "training.logit_adjustment_tau"),

        "class_balance": nested_get(config, "training.class_balance"),
        "monitor": nested_get(config, "training.monitor"),

        "best_epoch": best.get("best_epoch"),
        "best_score": to_float(best.get("best_score")),
        "best_monitor": best.get("monitor"),
    }

    for class_name, count in class_counts.items():
        row[f"class_count_{class_name}"] = count

    if val_metrics:
        add_metric_prefix(row, val_metrics, "val")

    if test_metrics:
        add_metric_prefix(row, test_metrics, "test")

    return row


def collect_runs(root: Path) -> pd.DataFrame:
    rows = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        row = collect_run(run_dir)

        if row is not None:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def make_summary_by_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_cols = ["method"]

    if "backbone" in df.columns:
        group_cols.append("backbone")

    metric_cols = [
        "test_macro_f1",
        "test_balanced_accuracy",
        "test_mcc",
        "test_roc_auc_macro_ovr",
        "test_pr_auc_macro",

        "test_calibration_after_ece",
        "test_calibration_after_brier",
        "test_calibration_after_nll",

        "test_embedding_silhouette",
        "test_embedding_intra_class_distance",
        "test_embedding_inter_class_distance",
        "test_embedding_inter_intra_ratio",

        "test_malignant_mcc_sensitivity",
        "test_malignant_mcc_specificity",
        "test_malignant_mcc_f1",
        "test_malignant_mcc_mcc",
        "test_malignant_mcc_threshold",

        "test_melanoma_mcc_sensitivity",
        "test_melanoma_mcc_specificity",
        "test_melanoma_mcc_f1",
        "test_melanoma_mcc_mcc",
        "test_melanoma_mcc_threshold",
    ]

    metric_cols = [c for c in metric_cols if c in df.columns]

    if not metric_cols:
        return pd.DataFrame()

    summary = (
        df
        .groupby(group_cols, dropna=False)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default="runs",
        help="Root directory with experiment runs.",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path for per-run table. Default: <root>/summary_per_run.csv",
    )

    parser.add_argument(
        "--summary-out",
        default=None,
        help="Output CSV path for method summary. Default: <root>/summary_by_method.csv",
    )

    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Runs root does not exist: {root}")

    per_run_out = Path(args.out) if args.out else root / "summary_per_run.csv"
    summary_out = (
        Path(args.summary_out)
        if args.summary_out
        else root / "summary_by_method.csv"
    )

    df = collect_runs(root)

    if df.empty:
        print(f"No runs found in {root}")
        return

    df = df.sort_values(
        by=[
            col for col in ["method", "seed", "run_name"]
            if col in df.columns
        ]
    )

    df.to_csv(per_run_out, index=False)

    summary = make_summary_by_method(df)

    if not summary.empty:
        summary.to_csv(summary_out, index=False)

    print(f"Saved per-run table: {per_run_out}")
    print(f"Rows: {len(df)}")

    if not summary.empty:
        print(f"Saved method summary: {summary_out}")
        print(f"Methods: {df['method'].nunique() if 'method' in df.columns else 'unknown'}")


if __name__ == "__main__":
    main()
