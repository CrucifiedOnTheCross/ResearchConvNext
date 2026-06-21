from __future__ import annotations

import argparse
import gc
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ham_pipeline.data import CLASSES, make_loaders
from ham_pipeline.config import load_config, set_config_value
from ham_pipeline.classification_losses import (
    ClassificationLossConfig,
    build_classification_loss,
)
from ham_pipeline.losses import build_metric_loss
from ham_pipeline.methods import MethodSpec, all_methods, get_method
from ham_pipeline.metrics import evaluate_arrays, save_plots
from ham_pipeline.model import ConvNeXtMetric
from ham_pipeline.utils import *


def autocast_ctx(precision: str):
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    return torch.amp.autocast(
        "cuda",
        dtype=dtype,
        enabled=precision in ("bf16", "fp16"),
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def get_proxies(model: torch.nn.Module) -> torch.nn.Parameter | None:
    raw = unwrap_model(model)
    return raw.proxies if hasattr(raw, "proxies") else None


def build_model(cfg: dict, device: torch.device) -> ConvNeXtMetric:
    model = ConvNeXtMetric(
        cfg["name"],
        len(CLASSES),
        cfg["embedding_dim"],
        cfg["dropout"],
        cfg["pretrained"],
        cfg.get("angular_scale", 30.0),
        cfg.get("angular_margin", 0.2),
    )

    model.set_checkpointing(cfg.get("gradient_checkpointing", False))

    return model.to(device)


def _clear_cuda(device: torch.device) -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def _is_cuda_oom(error: BaseException) -> bool:
    message = str(error).lower()

    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        "out of memory" in message and ("cuda" in message or "cudnn" in message)
    )


def _probe_batch(
    cfg: dict,
    device: torch.device,
    batch_size: int,
    method: MethodSpec,
) -> float:
    tc = cfg["training"]
    mc = cfg["model"]
    size = cfg["data"]["image_size"]

    model = build_model(mc, device).train()

    multiplier = 2 if method.two_views else 1

    images = torch.randn(
        batch_size * multiplier,
        3,
        size,
        size,
        device=device,
    )

    if tc["channels_last"]:
        model = model.to(memory_format=torch.channels_last)
        images = images.to(memory_format=torch.channels_last)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    with autocast_ctx(tc["precision"]):
        labels = torch.arange(len(images), device=device) % len(CLASSES)
        need_embedding = method.metric_loss is not None or method.classifier_mode != "linear"
        logits, embeddings = model(
            images,
            labels=labels if method.classifier_mode != "linear" else None,
            classifier_mode=method.classifier_mode,
            compute_embedding=need_embedding,
        )
        loss = logits.mean()
        if embeddings is not None:
            loss = loss + embeddings.mean()

    loss.backward()
    optimizer.step()

    torch.cuda.synchronize(device)

    return torch.cuda.max_memory_allocated(device) / (1024**3)


def find_batch(
    cfg: dict,
    device: torch.device,
    logger,
    method: MethodSpec,
) -> int:
    tc = cfg["training"]

    if tc["batch_size"] != "auto":
        return int(tc["batch_size"])

    for batch_size in tc["batch_candidates"]:
        _clear_cuda(device)

        try:
            peak_vram = _probe_batch(
                cfg=cfg,
                device=device,
                batch_size=batch_size,
                method=method,
            )
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise

            logger.info(
                "Batch %s OOM; trying a smaller candidate",
                batch_size,
            )

            _clear_cuda(device)
            continue

        _clear_cuda(device)

        logger.info(
            "Auto batch selected: %s | two_views=%s | probe peak %.2f GiB",
            batch_size,
            method.two_views,
            peak_vram,
        )

        return batch_size

    raise RuntimeError("No batch candidate fits VRAM")


def class_counts_from_frame(frame) -> torch.Tensor:
    train_frame = frame[frame.split == "train"]
    counts = train_frame.dx.value_counts()

    return torch.tensor(
        [int(counts.get(class_name, 0)) for class_name in CLASSES],
        dtype=torch.float32,
    ).clamp_min(1.0)


def run_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None,
    classification_loss: torch.nn.Module,
    metric_loss: torch.nn.Module | None,
    method: MethodSpec,
    scaler,
    device: torch.device,
    cfg: dict,
    class_counts: torch.Tensor,
    train: bool = True,
    epoch: int = 0,
    epochs: int = 0,
) -> dict:
    tc = cfg["training"]

    model.train(train)

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    total_loss = torch.zeros((), device=device)
    total_cls_loss = torch.zeros((), device=device)
    total_metric_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    n = 0

    accum = max(1, math.ceil(tc["effective_batch_size"] / loader.batch_size))

    start = time.perf_counter()
    updates = max(1, int(tc.get("progress_updates", 20)))

    context = torch.enable_grad if train else torch.inference_mode
    desc = f"Train {epoch:02d}/{epochs:02d}" if train else "Evaluate"

    batches = tqdm(
        loader,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=train,
        disable=not tc.get("progress_bar", True),
    )

    with context():
        for step, (x, y, _) in enumerate(batches):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if x.ndim == 5:
                views = x.shape[1]
                x = x.flatten(0, 1)
                y = y.repeat_interleave(views)

            if tc["channels_last"]:
                x = x.to(memory_format=torch.channels_last)

            with autocast_ctx(tc["precision"]):
                logits, z = model(
                    x,
                    labels=y if method.classifier_mode != "linear" else None,
                    classifier_mode=method.classifier_mode,
                    compute_embedding=metric_loss is not None,
                    margin=tc.get("margin"),
                    scale=tc.get("angular_scale"),
                )

                cls = classification_loss(logits, y)
                combined = tc.get("ce_weight", 1.0) * cls
                metric = None
                if metric_loss is not None:
                    if z is None:
                        raise RuntimeError(f"Method {method.name} requires embeddings")
                    metric = metric_loss(
                        z,
                        y,
                        proxies=get_proxies(model),
                        class_counts=class_counts,
                    )
                    combined = combined + tc.get("metric_weight", 1.0) * metric
                loss = combined / accum

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum == 0 or step + 1 == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        tc["grad_clip"],
                    )

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

            batch_size = len(y)

            total_loss += loss.detach() * accum * batch_size
            total_cls_loss += cls.detach() * batch_size
            if metric is not None:
                total_metric_loss += metric.detach() * batch_size
            correct += (logits.argmax(dim=1) == y).sum()
            n += batch_size

            if (step + 1) % updates == 0 or step + 1 == len(loader):
                postfix = dict(
                    loss=f"{(total_loss / n).item():.3f}",
                    cls=f"{(total_cls_loss / n).item():.3f}",
                    acc=f"{100 * correct.item() / n:.1f}%",
                    ips=f"{n / (time.perf_counter() - start):.0f}",
                    vram=f"{torch.cuda.memory_allocated(device) / (1024**3):.1f}G",
                )
                if metric_loss is not None:
                    postfix["metric"] = f"{(total_metric_loss / n).item():.3f}"
                batches.set_postfix(**postfix)

    result = {
        "loss": float((total_loss / n).item()),
        "classification_loss": float((total_cls_loss / n).item()),
        "accuracy": float(correct.item() / n),
        "images_per_sec": float(n / (time.perf_counter() - start)),
    }
    if metric_loss is not None:
        result["metric_loss"] = float((total_metric_loss / n).item())
    return result


@torch.inference_mode()
def predict(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    method: MethodSpec,
    desc: str = "Evaluate",
):
    model.eval()

    logits_list = []
    labels_list = []
    embeddings_list = []
    ids = []

    batches = tqdm(
        loader,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        disable=not cfg["training"].get("progress_bar", True),
    )

    for x, y, image_ids in batches:
        x = x.to(device, non_blocking=True)

        if cfg["training"]["channels_last"]:
            x = x.to(memory_format=torch.channels_last)

        with autocast_ctx(cfg["training"]["precision"]):
            embedding_source = (
                "projector"
                if method.metric_loss is not None or method.classifier_mode != "linear"
                else "backbone"
            )
            logits, z = model(
                x,
                classifier_mode=method.classifier_mode,
                compute_embedding=True,
                embedding_source=embedding_source,
                margin=cfg["training"].get("margin"),
                scale=cfg["training"].get("angular_scale"),
            )
            if z is None:
                raise RuntimeError("Evaluation requires an embedding")

        logits_list.append(logits.float().cpu())
        labels_list.append(y)
        embeddings_list.append(z.float().cpu())
        ids.extend(image_ids)

    return (
        torch.cat(logits_list).numpy(),
        torch.cat(labels_list).numpy(),
        torch.cat(embeddings_list).numpy(),
        ids,
    )


def copy_split_summary(cfg: dict, run_dir: Path) -> None:
    candidates = [
        Path(cfg["data"]["root"]) / "split_summary.json",
        Path(cfg["data"]["splits"]).parent / "split_summary.json",
    ]

    for path in candidates:
        if path.exists():
            shutil.copy2(path, run_dir / "split_summary.json")
            return


def create_checkpoint_alias(source: Path, alias: Path) -> None:
    alias.unlink(missing_ok=True)
    try:
        os.link(source, alias)
    except OSError:
        try:
            alias.symlink_to(source.name)
        except OSError:
            if alias.name == "best.pt":
                shutil.copy2(source, alias)


def remove_unreferenced_checkpoints(run: Path, best_records: dict) -> None:
    referenced = {record["checkpoint"] for record in best_records.values()}
    for checkpoint in run.glob("checkpoint_epoch_*.pt"):
        if checkpoint.name not in referenced:
            checkpoint.unlink()


def evaluate_final_splits(
    raw_model: torch.nn.Module,
    loaders: dict,
    device: torch.device,
    cfg: dict,
    run: Path,
    logger,
    method: MethodSpec,
) -> None:
    """
    Final evaluation protocol.

    Validation:
    - fits temperature scaling
    - fits malignant/melanoma thresholds

    Test:
    - reuses validation temperature
    - reuses validation thresholds

    This prevents test leakage.
    """
    calibrated_temperature = None
    threshold_state = None

    for split in ("val", "test"):
        logits, y, z, ids = predict(
            raw_model,
            loaders[split],
            device,
            cfg,
            method,
            desc=split.capitalize(),
        )

        is_validation = split == "val"

        metrics, probs, pred, used_temperature = evaluate_arrays(
            logits,
            y,
            z,
            CLASSES,
            temperature=calibrated_temperature,
            fit_temperature=is_validation,
            threshold_state=threshold_state,
            fit_thresholds_on_this_split=is_validation,
        )

        if is_validation:
            calibrated_temperature = used_temperature
            threshold_state = metrics.get("threshold_state")

            save_json(
                {
                    "temperature": calibrated_temperature,
                    "threshold_state": threshold_state,
                },
                run / "validation_calibration_and_thresholds.json",
            )
        else:
            save_json(
                {
                    "temperature_from_validation": calibrated_temperature,
                    "threshold_state_from_validation": threshold_state,
                },
                run / "test_used_calibration_and_thresholds.json",
            )

        save_json(metrics, run / f"{split}_metrics.json")

        np.savez_compressed(
            run / f"{split}_predictions.npz",
            ids=np.array(ids),
            y=y,
            logits=logits,
            probs=probs,
            embeddings=z,
        )

        if cfg["output"]["visualizations"]:
            save_plots(
                y,
                pred,
                z,
                CLASSES,
                run,
                prefix=split,
            )

        logger.info(
            "%s metrics | macro_f1=%.4f | balanced_accuracy=%.4f | mcc=%.4f | ece_after=%.4f",
            split,
            metrics["macro_f1"],
            metrics["balanced_accuracy"],
            metrics["mcc"],
            metrics["calibration_after"]["ece"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path)

    args = parser.parse_args()

    eval_checkpoint = None
    if args.eval_only:
        if args.checkpoint is None:
            raise ValueError("--eval-only requires --checkpoint")
        eval_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = eval_checkpoint.get("config", load_config(args.config))
    else:
        cfg = load_config(args.config)

    for item in args.set:
        set_config_value(cfg, *item.split("=", 1))

    method = cfg["training"]["method"]

    method_spec = get_method(method)
    cfg["training"]["approximate_method"] = method_spec.approximate

    seed_everything(cfg["seed"])
    setup_runtime()

    if args.eval_only:
        run = args.checkpoint.resolve().parent
    else:
        run = make_run_dir(cfg["output"]["root"], method)
    logger = configure_logging(run)
    writer = None if args.eval_only else SummaryWriter(run / "tensorboard")

    if not args.eval_only:
        (run / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False),
            encoding="utf-8",
        )
        save_json(system_info(), run / "system.json")
        copy_split_summary(cfg, run)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda")

    two_views = method_spec.two_views

    logger.info(
        "Method=%s | two_views=%s | class_balance=%s",
        method,
        two_views,
        cfg["training"].get("class_balance"),
    )
    if method_spec.approximate:
        logger.warning(
            "%s is a compact experimental adaptation, not an official paper reproduction",
            method,
        )

    batch = (
        int(cfg.get("evaluation", {}).get("batch_size", 64))
        if args.eval_only
        else find_batch(cfg=cfg, device=device, logger=logger, method=method_spec)
    )

    loaders, frame = make_loaders(
        cfg["data"]["splits"],
        cfg["data"]["image_size"],
        batch,
        cfg["training"]["workers"],
        cfg["training"]["prefetch_factor"],
        cfg["training"]["class_balance"],
        False if args.eval_only else two_views,
        seed=cfg["seed"],
    )

    class_counts = class_counts_from_frame(frame)
    class_counts = class_counts.to(device)

    save_json(
        {
            class_name: int(class_counts[i].item())
            for i, class_name in enumerate(CLASSES)
        },
        run / "class_counts.json",
    )

    model = build_model(cfg["model"], device)

    if cfg["training"]["channels_last"]:
        model = model.to(memory_format=torch.channels_last)

    raw_model = model

    if args.eval_only:
        raw_model.load_state_dict(eval_checkpoint["model"])
        evaluate_final_splits(
            raw_model=raw_model,
            loaders=loaders,
            device=device,
            cfg=cfg,
            run=run,
            logger=logger,
            method=method_spec,
        )
        logger.info("Evaluation finished. Artifacts: %s", run.resolve())
        return

    if cfg["model"]["compile"]:
        try:
            model = torch.compile(
                model,
                mode="max-autotune-no-cudagraphs",
            )
            logger.info("torch.compile enabled")
        except Exception as error:
            logger.warning("torch.compile unavailable: %s", error)

    tc = cfg["training"]

    backbone_params = [
        p for name, p in raw_model.named_parameters()
        if name.startswith("backbone")
    ]

    head_params = [
        p for name, p in raw_model.named_parameters()
        if not name.startswith("backbone")
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": tc["learning_rate"],
            },
            {
                "params": head_params,
                "lr": tc["head_learning_rate"],
            },
        ],
        weight_decay=tc["weight_decay"],
    )

    warmup_epochs = max(1, tc["warmup_epochs"])
    epochs = tc["epochs"]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: (
            (e + 1) / warmup_epochs
            if e < warmup_epochs
            else 0.5
            * (
                1
                + math.cos(
                    math.pi
                    * (e - warmup_epochs)
                    / max(1, epochs - warmup_epochs)
                )
            )
        ),
    )

    classification_loss = build_classification_loss(
        ClassificationLossConfig(
            name=method_spec.classification_loss,
            label_smoothing=tc.get("label_smoothing", 0.0),
            focal_gamma=tc.get("focal_gamma", 2.0),
            focal_use_class_weights=tc.get("focal_use_class_weights", False),
            logit_adjustment_tau=tc.get("logit_adjustment_tau", 1.0),
        ),
        class_counts=class_counts.detach().cpu(),
    ).to(device)

    metric_loss = build_metric_loss(
        method_spec.metric_loss,
        tc.get("temperature", 0.1),
        tc.get("margin", 0.2),
    )
    if metric_loss is not None:
        metric_loss = metric_loss.to(device)

    scaler = (
        torch.amp.GradScaler("cuda")
        if tc["precision"] == "fp16"
        else None
    )

    monitors = list(dict.fromkeys(tc.get("checkpoint_monitors", [tc["monitor"]])))
    if tc["monitor"] not in monitors:
        monitors.insert(0, tc["monitor"])
    best_scores = {monitor: -float("inf") for monitor in monitors}
    best_records = {}
    bad = 0
    history = []

    for epoch in range(epochs):
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            classification_loss=classification_loss,
            metric_loss=metric_loss,
            method=method_spec,
            scaler=scaler,
            device=device,
            cfg=cfg,
            class_counts=class_counts,
            train=True,
            epoch=epoch + 1,
            epochs=epochs,
        )

        val_logits, val_y, val_z, _ = predict(
            model,
            loaders["val"],
            device,
            cfg,
            method_spec,
            desc=f"Valid {epoch + 1:02d}/{epochs:02d}",
        )

        val_metrics, _, _, _ = evaluate_arrays(
            val_logits,
            val_y,
            val_z,
            CLASSES,
        )

        scheduler.step()

        row = {
            "epoch": epoch + 1,
            **{
                f"train_{key}": value
                for key, value in train_metrics.items()
            },
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_mcc": float(val_metrics["mcc"]),
            "val_malignant_mcc": float(val_metrics["binary_malignant"]["mcc"]),
            "val_malignant_sensitivity": float(val_metrics["binary_malignant"]["sensitivity"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }

        history.append(row)

        logger.info("epoch %s: %s", epoch + 1, row)

        for key, value in row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(key, value, epoch + 1)

        primary_improved = False
        improved_monitors = []
        for monitor in monitors:
            if monitor not in row:
                raise ValueError(f"Unknown checkpoint monitor {monitor!r}; row has {sorted(row)}")
            score = float(row[monitor])
            if score <= best_scores[monitor]:
                continue
            best_scores[monitor] = score
            improved_monitors.append(monitor)
            if monitor == tc["monitor"]:
                primary_improved = True

        if improved_monitors:
            checkpoint_name = f"checkpoint_epoch_{epoch + 1:03d}.pt"
            payload = {
                "model": raw_model.state_dict(),
                "config": cfg,
                "epoch": epoch + 1,
                "method": method,
                "class_counts": class_counts.detach().cpu(),
                "scores": {key: float(row[key]) for key in monitors},
            }
            torch.save(payload, run / checkpoint_name)
            for monitor in improved_monitors:
                best_records[monitor] = {
                    "epoch": epoch + 1,
                    "score": float(row[monitor]),
                    "checkpoint": checkpoint_name,
                }
            remove_unreferenced_checkpoints(run, best_records)
        bad = 0 if primary_improved else bad + 1

        save_json(history, run / "history.json")

        if bad >= tc["early_stopping"]:
            logger.info("Early stopping")
            break

    primary_record = best_records[tc["monitor"]]
    primary_checkpoint = run / primary_record["checkpoint"]
    checkpoint = torch.load(
        primary_checkpoint,
        map_location=device,
        weights_only=False,
    )

    create_checkpoint_alias(primary_checkpoint, run / "best.pt")
    for monitor, record in best_records.items():
        alias = run / f"best_{monitor.removeprefix('val_')}.pt"
        create_checkpoint_alias(run / record["checkpoint"], alias)

    raw_model.load_state_dict(checkpoint["model"])

    save_json(
        {
            "best_epoch": checkpoint["epoch"],
            "best_score": primary_record["score"],
            "monitor": tc["monitor"],
            "method": checkpoint["method"],
            "all_checkpoints": best_records,
        },
        run / "best_summary.json",
    )

    evaluate_final_splits(
        raw_model=raw_model,
        loaders=loaders,
        device=device,
        cfg=cfg,
        run=run,
        logger=logger,
        method=method_spec,
    )

    writer.close()

    logger.info("Finished. Artifacts: %s", run.resolve())


if __name__ == "__main__":
    main()
