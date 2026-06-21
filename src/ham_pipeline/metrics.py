from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def to_builtin(obj):
    """
    Converts numpy / torch scalar values to JSON-safe Python types.
    Prevents logs and JSON from containing np.float64(...).
    """
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]

    if isinstance(obj, tuple):
        return tuple(to_builtin(v) for v in obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()

    return obj


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)

        if mask.any():
            score += mask.mean() * abs(
                (pred[mask] == y[mask]).mean() - conf[mask].mean()
            )

    return float(score)


def temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float64)
    target = torch.tensor(y, dtype=torch.long)

    temperature = torch.ones(
        1,
        dtype=torch.float64,
        requires_grad=True,
    )

    optimizer = torch.optim.LBFGS(
        [temperature],
        lr=0.05,
        max_iter=80,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(
            logits_t / temperature.clamp(0.05, 10.0),
            target,
        )
        loss.backward()
        return loss

    optimizer.step(closure)

    return float(temperature.detach().clamp(0.05, 10.0).item())


def calibration(probs: np.ndarray, y: np.ndarray) -> dict:
    onehot = np.eye(probs.shape[1])[y]

    return {
        "ece": float(ece(probs, y)),
        "brier": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
        "nll": float(log_loss(y, probs, labels=list(range(probs.shape[1])))),
    }


def safe_binary_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[False, True],
    ).ravel()

    return int(tn), int(fp), int(fn), int(tp)


def binary_metrics_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y_pred = scores >= threshold

    tn, fp, fn, tp = safe_binary_confusion(y_true, y_pred)

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    out = {
        "threshold": float(threshold),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "mcc": float(
            matthews_corrcoef(y_true, y_pred)
        ),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except ValueError:
        out["roc_auc"] = None
        out["pr_auc"] = None

    return out


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    """
    Uses unique score values plus 0.5.
    This is enough for validation threshold search on small/medium datasets.
    """
    unique = np.unique(scores)

    if len(unique) > 2000:
        quantiles = np.linspace(0.0, 1.0, 2000)
        unique = np.unique(np.quantile(unique, quantiles))

    thresholds = np.unique(
        np.concatenate(
            [
                np.array([0.5], dtype=float),
                unique.astype(float),
            ]
        )
    )

    return np.clip(thresholds, 0.0, 1.0)


def fit_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """
    Fits thresholds by validation scores.

    Returned thresholds must then be reused on test.
    """
    thresholds = candidate_thresholds(scores)

    best = {
        "fixed_0_5": 0.5,
        "youden": 0.5,
        "mcc": 0.5,
        "f1": 0.5,
        "balanced_accuracy": 0.5,
    }

    best_scores = {
        "youden": -float("inf"),
        "mcc": -float("inf"),
        "f1": -float("inf"),
        "balanced_accuracy": -float("inf"),
    }

    for threshold in thresholds:
        y_pred = scores >= threshold
        tn, fp, fn, tp = safe_binary_confusion(y_true, y_pred)
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denominator) if denominator else 0.0
        ba = 0.5 * (sensitivity + specificity)
        youden = sensitivity + specificity - 1.0

        if youden > best_scores["youden"]:
            best_scores["youden"] = youden
            best["youden"] = float(threshold)

        if mcc > best_scores["mcc"]:
            best_scores["mcc"] = mcc
            best["mcc"] = float(threshold)

        if f1 > best_scores["f1"]:
            best_scores["f1"] = f1
            best["f1"] = float(threshold)

        if ba > best_scores["balanced_accuracy"]:
            best_scores["balanced_accuracy"] = ba
            best["balanced_accuracy"] = float(threshold)

    return best


def threshold_report(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: Optional[dict] = None,
) -> dict:
    if thresholds is None:
        thresholds = fit_thresholds(y_true, scores)

    out = {
        "thresholds": thresholds,
        "metrics": {},
    }

    for name, threshold in thresholds.items():
        out["metrics"][name] = binary_metrics_from_scores(
            y_true,
            scores,
            float(threshold),
        )

    return out


def binary_targets_and_scores(
    y: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
) -> dict:
    mel_idx = classes.index("mel")

    malignant_classes = ("mel", "bcc", "akiec")
    malignant_indices = [classes.index(c) for c in malignant_classes]

    malignant_true = np.isin(y, malignant_indices)
    malignant_scores = probs[:, malignant_indices].sum(axis=1)

    melanoma_true = y == mel_idx
    melanoma_scores = probs[:, mel_idx]

    return {
        "malignant": {
            "y_true": malignant_true,
            "scores": malignant_scores,
            "positive_classes": list(malignant_classes),
        },
        "melanoma": {
            "y_true": melanoma_true,
            "scores": melanoma_scores,
            "positive_classes": ["mel"],
        },
    }


def evaluate_thresholds(
    y: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    threshold_state: Optional[dict] = None,
    fit_thresholds_on_this_split: bool = False,
) -> tuple[dict, dict]:
    """
    Evaluates malignant and melanoma thresholds.

    If fit_thresholds_on_this_split=True, thresholds are fitted on this split.
    For correct final evaluation:
    - validation: fit_thresholds_on_this_split=True
    - test: pass threshold_state from validation
    """
    targets = binary_targets_and_scores(y, probs, classes)

    out = {}
    fitted_state = {} if threshold_state is None else dict(threshold_state)

    for task_name, task in targets.items():
        y_true = task["y_true"]
        scores = task["scores"]

        if fit_thresholds_on_this_split:
            thresholds = fit_thresholds(y_true, scores)
            fitted_state[task_name] = thresholds
        elif task_name not in fitted_state:
            thresholds = {"fixed_0_5": 0.5}
        else:
            thresholds = fitted_state[task_name]

        report = threshold_report(y_true, scores, thresholds)
        report["positive_classes"] = task["positive_classes"]

        out[task_name] = report

    return out, fitted_state


def embedding_metrics(
    z: np.ndarray,
    y: np.ndarray,
    classes: list[str],
) -> dict:
    out = {}

    if len(y) > 2 and len(y) <= 10000:
        try:
            out["silhouette"] = float(
                silhouette_score(
                    z,
                    y,
                    sample_size=min(5000, len(y)),
                    random_state=42,
                )
            )
        except ValueError:
            out["silhouette"] = None

    centers = []

    for i in range(len(classes)):
        mask = y == i

        if mask.any():
            centers.append(z[mask].mean(axis=0))
        else:
            centers.append(np.zeros(z.shape[1], dtype=z.dtype))

    centers = np.stack(centers)

    intra_values = []

    for i in range(len(classes)):
        mask = y == i

        if mask.any():
            intra_values.append(
                np.linalg.norm(z[mask] - centers[i], axis=1).mean()
            )

    intra = float(np.mean(intra_values)) if intra_values else 0.0

    inter = np.linalg.norm(
        centers[:, None] - centers[None, :],
        axis=2,
    )

    inter = inter[np.triu_indices(len(classes), 1)].mean()

    out.update(
        {
            "intra_class_distance": float(intra),
            "inter_class_distance": float(inter),
            "inter_intra_ratio": float(inter / max(intra, 1e-12)),
        }
    )

    return out


def multiclass_auc_metrics(
    y: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
) -> dict:
    out = {}

    try:
        out["roc_auc_macro_ovr"] = float(
            roc_auc_score(
                y,
                probs,
                multi_class="ovr",
                average="macro",
            )
        )

        out["pr_auc_macro"] = float(
            average_precision_score(
                np.eye(len(classes))[y],
                probs,
                average="macro",
            )
        )

        out["per_class_auc"] = {
            class_name: {
                "roc_auc": float(roc_auc_score(y == i, probs[:, i])),
                "pr_auc": float(average_precision_score(y == i, probs[:, i])),
            }
            for i, class_name in enumerate(classes)
        }
    except ValueError:
        pass

    return out


def evaluate_arrays(
    logits: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    classes: list[str],
    temperature: Optional[float] = None,
    fit_temperature: bool = False,
    threshold_state: Optional[dict] = None,
    fit_thresholds_on_this_split: Optional[bool] = None,
):
    """
    Main metric entry point.

    Backward-compatible return:
    metrics, probs, pred, used_temperature

    New behavior:
    - validation can fit thresholds and store them in metrics["threshold_state"]
    - test can receive threshold_state from validation
    """
    y = np.asarray(y)

    raw_probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    pred = raw_probs.argmax(axis=1)

    report = classification_report(
        y,
        pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "mcc": float(matthews_corrcoef(y, pred)),
        "per_class": {
            class_name: {
                key: float(report[class_name][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for class_name in classes
        },
    }

    out.update(multiclass_auc_metrics(y, raw_probs, classes))

    out["calibration_before"] = calibration(raw_probs, y)

    used_temperature = (
        temperature_scale(logits, y)
        if fit_temperature
        else (temperature or 1.0)
    )

    scaled_probs = torch.softmax(
        torch.tensor(logits) / used_temperature,
        dim=1,
    ).numpy()

    out["temperature_from_validation"] = float(used_temperature)
    out["calibration_after"] = calibration(scaled_probs, y)

    # For compatibility, the returned probs are calibrated/scaled probabilities.
    probs = scaled_probs
    pred = probs.argmax(axis=1)

    # Threshold fitting defaults:
    # If we fit temperature on validation, we also fit thresholds on validation.
    # For test, train.py should pass threshold_state from validation.
    if fit_thresholds_on_this_split is None:
        fit_thresholds_on_this_split = fit_temperature

    threshold_metrics, fitted_threshold_state = evaluate_thresholds(
        y=y,
        probs=probs,
        classes=classes,
        threshold_state=threshold_state,
        fit_thresholds_on_this_split=fit_thresholds_on_this_split,
    )

    out["threshold_analysis"] = threshold_metrics
    out["threshold_state"] = fitted_threshold_state

    # Backward-compatible top-level binary_malignant at fixed 0.5.
    out["binary_malignant"] = threshold_metrics["malignant"]["metrics"]["fixed_0_5"]

    # More explicit aliases for common reporting.
    out["binary_malignant_thresholded"] = threshold_metrics["malignant"]["metrics"]
    out["binary_melanoma_thresholded"] = threshold_metrics["melanoma"]["metrics"]

    out["embedding"] = embedding_metrics(z, y, classes)

    return to_builtin(out), probs, pred, float(used_temperature)


def save_plots(
    y: np.ndarray,
    pred: np.ndarray,
    z: np.ndarray,
    classes: list[str],
    outdir: Path,
    prefix: str = "test",
) -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y, pred, normalize="true")

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        xticklabels=classes,
        yticklabels=classes,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outdir / f"{prefix}_confusion_matrix.png", dpi=180)
    plt.close()

    try:
        import umap
        from sklearn.manifold import TSNE

        emb = umap.UMAP(
            n_neighbors=20,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        ).fit_transform(z)

        plt.figure(figsize=(9, 7))

        for i, class_name in enumerate(classes):
            plt.scatter(
                *emb[y == i].T,
                s=8,
                alpha=0.65,
                label=class_name,
            )

        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_umap.png", dpi=180)
        plt.close()

        np.save(outdir / f"{prefix}_umap.npy", emb)

        tsne = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=min(30, max(5, len(y) // 20)),
            random_state=42,
        ).fit_transform(z)

        plt.figure(figsize=(9, 7))

        for i, class_name in enumerate(classes):
            plt.scatter(
                *tsne[y == i].T,
                s=8,
                alpha=0.65,
                label=class_name,
            )

        plt.legend(markerscale=2)
        plt.tight_layout()
        plt.savefig(outdir / f"{prefix}_tsne.png", dpi=180)
        plt.close()

        np.save(outdir / f"{prefix}_tsne.npy", tsne)

    except Exception as error:
        (outdir / f"{prefix}_embedding_plot_error.txt").write_text(
            str(error),
            encoding="utf-8",
        )
